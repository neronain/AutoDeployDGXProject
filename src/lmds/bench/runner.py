"""วัดความเร็วจริงของโมเดลผ่าน OpenAI API — ไม่พึ่ง `bench` ของ controller

ทำไมไม่ใช้ controller: คำสั่ง `bench` มีไม่ครบทุก engine (bundle llama.cpp ที่สร้างจาก
เทมเพลตปัจจุบันไม่มีเลย) และแต่ละตัวก็วัดคนละวิธี — ตัวเลขที่ออกมาจึงเทียบข้าม engine
ไม่ได้ ซึ่งเป็นเรื่องเดียวที่คนอยากทำกับหน้าคะแนน

วิธีวัดที่ใช้ (ตรงกับที่เครื่องมือ benchmark ทั่วไปทำ):
  TTFT      เวลาตั้งแต่ส่งคำขอจนได้ token แรก — รวม queue + prefill
  decode    (token ที่ได้ − 1) ÷ เวลาตั้งแต่ token แรกถึง token สุดท้าย
            ลบหนึ่งเพราะ token แรกถูกนับไปแล้วใน TTFT การไม่ลบทำให้ตัวเลขสูงเกินจริง
  prefill   prompt_tokens ÷ TTFT — เป็นค่าประมาณ ไม่ใช่ prefill ล้วน (มี queue ปนอยู่)
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field

import httpx

from .workloads import Workload


class BenchError(Exception):
    pass


@dataclass
class Sample:
    """ผลยิงหนึ่งครั้ง"""

    ttft_s: float
    decode_s: float
    total_s: float
    prompt_tokens: int
    completion_tokens: int
    # token ที่เซิร์ฟเวอร์บอกว่าใช้ของเก่าจาก cache — ควรเป็น 0 ถ้า nonce ทำงาน
    # ไม่เป็น 0 แปลว่าตัวเลข TTFT/prefill รอบนี้เชื่อไม่ได้
    cached_tokens: int = 0

    @property
    def decode_tps(self) -> float:
        # token แรกอยู่ใน TTFT แล้ว — ช่วง decode จึงมี token ที่เหลือเท่านั้น
        produced = max(0, self.completion_tokens - 1)
        return produced / self.decode_s if self.decode_s > 0 and produced else 0.0

    @property
    def prefill_tps(self) -> float:
        return self.prompt_tokens / self.ttft_s if self.ttft_s > 0 else 0.0


@dataclass
class WorkloadResult:
    key: str
    label: str
    target_input: int
    samples: list[Sample] = field(default_factory=list)
    error: str = ""

    @property
    def cache_hits(self) -> int:
        return sum(s.cached_tokens for s in self.samples)

    def _median(self, attr: str) -> float:
        values = [getattr(s, attr) for s in self.samples]
        return round(statistics.median(values), 3) if values else 0.0

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "target_input": self.target_input,
            "runs": len(self.samples),
            "error": self.error,
            # median ไม่ใช่ mean — ยิงครั้งแรกมักช้ากว่าเพราะ cache ยังไม่อุ่น
            # ค่าเฉลี่ยจะถูกครั้งเดียวนั้นดึงลงทั้งชุด
            "ttft_s": self._median("ttft_s"),
            "decode_tps": self._median("decode_tps"),
            "prefill_tps": self._median("prefill_tps"),
            "total_s": self._median("total_s"),
            "prompt_tokens": int(self._median("prompt_tokens")),
            "completion_tokens": int(self._median("completion_tokens")),
            # >0 แปลว่า prefix cache ยังกินอยู่ — TTFT/prefill ของรอบนี้ต่ำกว่าความจริง
            "cached_tokens": self.cache_hits,
        }


def _stream_once(client: httpx.Client, endpoint: str, model: str,
                 workload: Workload, timeout: float, nonce: str) -> Sample:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": workload.prompt(nonce)}],
        "max_tokens": workload.output_tokens,
        "stream": True,
        # ขอ usage ท้ายสตรีม — ไม่งั้นต้องนับ chunk เอง ซึ่งนับ token ไม่ตรงกับ tokenizer
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token_at = 0.0
    last_token_at = 0.0
    chunk_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0

    with client.stream("POST", f"{endpoint}/chat/completions", json=body, timeout=timeout) as response:
        if response.status_code != 200:
            response.read()
            raise BenchError(f"HTTP {response.status_code}: {response.text[:200]}")
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = event.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
                completion_tokens = usage.get("completion_tokens") or completion_tokens
                details = usage.get("prompt_tokens_details") or {}
                cached_tokens = details.get("cached_tokens") or cached_tokens
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                # โมเดลสาย reasoning ส่งความคิดมาก่อนคำตอบ — token พวกนั้นก็ถูก decode
                # เหมือนกัน ไม่นับ = บอกว่าโมเดลช้ากว่าความจริงมาก
                produced = delta.get("content") or delta.get("reasoning_content")
                if not produced:
                    continue
                now = time.perf_counter()
                if not first_token_at:
                    first_token_at = now
                last_token_at = now
                chunk_tokens += 1

    ended = time.perf_counter()
    if not first_token_at:
        raise BenchError("ไม่ได้ token สักตัวจากเซิร์ฟเวอร์")
    return Sample(
        ttft_s=first_token_at - started,
        decode_s=max(0.0, (last_token_at or ended) - first_token_at),
        total_s=ended - started,
        prompt_tokens=prompt_tokens,
        # เซิร์ฟเวอร์ที่ไม่ส่ง usage มา — ใช้จำนวน chunk แทน (หยาบกว่า แต่ดีกว่าศูนย์)
        completion_tokens=completion_tokens or chunk_tokens,
        cached_tokens=cached_tokens,
    )


def measure(endpoint: str, model: str, workloads, runs: int = 3,
            timeout: float = 300.0, api_key: str = "",
            on_progress=None) -> list[WorkloadResult]:
    """ยิงทุก workload ตามจำนวนรอบที่กำหนด แล้วคืนผลแบบ median

    ยิงรอบอุ่นเครื่องหนึ่งครั้งต่อ workload โดยไม่นับผล — ครั้งแรกของแต่ละความยาว
    ต้องจัดสรร KV cache ใหม่ ซึ่งช้ากว่ารอบถัด ๆ ไปอย่างเห็นได้ชัด
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    results: list[WorkloadResult] = []
    with httpx.Client(headers=headers) as client:
        for workload in workloads:
            result = WorkloadResult(workload.key, workload.label, workload.input_tokens)
            try:
                _stream_once(client, endpoint, model, workload, timeout, "warmup")
                for index in range(runs):
                    if on_progress:
                        on_progress(workload, index + 1, runs)
                    # nonce ต่างกันทุกรอบ — prefix cache จึงใช้ต่อไม่ได้
                    result.samples.append(_stream_once(
                        client, endpoint, model, workload, timeout, f"{workload.key}-{index}"))
            except (BenchError, httpx.HTTPError) as exc:
                result.error = str(exc)[:300]
            results.append(result)
    return results
