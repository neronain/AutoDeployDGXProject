# DGX Spark (GB10 / SM121) — vLLM Serving Field Notes

บันทึกความรู้ภาคสนามสำหรับ deploy โมเดลบน DGX Spark ที่ **สกัดจาก community repo หลายเจ้า**
(ทำ inference บน GB10/SM121 จริง) มาไว้ที่เดียว เพื่อใช้ตอนตั้ง recipe / เขียน controller / debug

> **ระดับความเชื่อถือ** — ทำเครื่องหมายไว้ทุกข้อ:
> - ✅ **corroborated** = เจอตรงกัน≥2 แหล่งอิสระ หรือเป็น hardware fact → เชื่อได้
> - ⚠️ **vendor-specific** = มาจาก image/tree/patch ของเจ้าเดียว, flag/PR# เป็นของ fork เขา →
>   ใช้เป็นทิศทาง ต้อง **verify กับ image ของเราเอง** ก่อน (อย่า copy flag ดิบ)
>
> กฎที่ robust พอถูกย้ายไปเป็น `arch_notes()` / `arch_requirements()` ใน `brain/rulebased.py` แล้ว
> (ดู `tests/test_arch_notes.py`) — เอกสารนี้คือรายละเอียดลึกที่ไม่ควรฝังใน rule engine

**แหล่ง (external, review-only — ไม่ใช่ของ neronain):**
- `eugr/spark-vllm-docker` — recipe-based vLLM deploy (recipe schema, patch catalog, cluster)
- `MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark` — DeepSeek-V4 บน 2× Spark (Anemll image)
- `AEON-7/vllm-ultimate-dgx-spark` — vLLM optimize + benchmark (DFlash, NVFP4-KV, quant gate)
- `albond/DGX_Spark_Unsloth_Lossless_Speedup` — **training** (Unsloth) — MTP head training

---

## 1. Hardware facts (SM121 / GB10) — ✅ corroborated

- **ไม่มี FP4 CUTLASS kernel ในตัว** → NVFP4 fallback เป็น Marlin SM80 ช้ามาก (~-42%) เว้นแต่ image มี kernel เฉพาะ
- **ไม่มี wgmma / tcgen05 / TMA multicast / DSMEM / cluster>1×1×1** (Hopper/SM100 เท่านั้น) — ยืนยันโดย NVIDIA staff (albond "What didn't ship")
- **Hardware FP8 e4m3 MMA มีจริง** แต่ end-to-end ที่ batch=1 launch overhead กินกำไรหมด (albond) → fp8 ไม่ค่อยคุ้มบน single stream
- **Unified LPDDR5X แชร์ CPU+GPU** → `--gpu-memory-utilization` ต้องต่ำ: AEON ใช้ ≤0.88, MiaAI ใช้ 0.835 (text)/0.80 (vision) กัน page-thrash
- **NVML ไม่รองรับ unified memory** → อย่าเปิด `VLLM_ENABLE_STARTUP_PLAN=1` (engine ตายตอน init) ⚠️ AEON เดียว
- **ไม่มี NVLink ระหว่าง Spark** — TP ข้ามเครื่องวิ่งบน ConnectX RoCE/IB เท่านั้น; อย่าเปิด NCCL symmetric memory ⚠️
- **build image เอง**: pin torch/torchvision/torchaudio วันเดียวกันทั้ง 2 build stage + `TORCH_CUDA_ARCH_LIST=12.1a`
  ไม่งั้น `undefined symbol …getCurrentCUDABlasHandle…` — ✅ ยืนยัน 2 แหล่ง (eugr Dockerfile:95,272 + memory เดิม)

## 2. NVFP4 มี **สองกับดักคนละเรื่อง** — ✅ corroborated

1. **Weight quant → Marlin fallback** (กฎเดิมของเรา): trigger จริงคือ **checkpoint ไม่มี per-input global scales**
   → บังคับ `--linear-backend flashinfer_cutlass` บน checkpoint ที่ไม่มี scale = crash (`input_global_scale_inv` หาย) ⚠️ AEON
2. **NVFP4-KV cache** (`nvfp4_ds_mla`): บน SM121 default layout = **NHD** แต่ upstream splitter ถูกเฉพาะ HND
   → **KV เพี้ยนเงียบๆ** เว้นแต่ image ใช้ Triton NVFP4-KV path (PR#44389/#44455) ⚠️ AEON `nvfp4_kv_gate.py`

**NVFP4 MoE backend เลือกตามโมเดล** (✅ eugr+AEON): `marlin` (default, ช้า) / `cutlass` (Nemotron-Nano/Super, gpt-oss)
/ `b12x` (DeepSeek-V4-0731, GLM-5.2). cutlass/b12x = path ที่เลี่ยง Marlin ได้ (ต้องมี image ที่มี kernel).
ทุก Marlin recipe ตั้ง `VLLM_MARLIN_USE_ATOMIC_ADD=1`.

**KV-cache dtype บน SM121** (✅ AEON): fp8 กับ nvfp4 KV throughput เท่ากัน (±1%) — nvfp4-kv **ไม่**ได้เร็วกว่า
แต่ได้ ~3× KV capacity (บล็อกมากขึ้น). **ข้อจำกัด**: nvfp4-kv ใช้ได้เฉพาะ **causal speculator** (mtp/eagle3/ngram)
→ ใช้กับ DFlash (non-causal) **ไม่ได้**.

## 3. Speculative decoding zoo — ✅ corroborated (หลาย method)

| method | โมเดล | k | ต้องมี draft repo แยก? | หมายเหตุ |
|---|---|---|---|---|
| `mtp` | Qwen3.5/3.6, Gemma4 | 2–3 | ไม่ (head ในตัว) | bit-exact @temp=0 · single Spark ~2-3× |
| `dspark` | DeepSeek-V4-0731, Nemotron-3.5-Lightning | 5–7 | ✅ (γ=5 rank-256 Markov head) | k ต้อง ≥ dspark_block_size=5 |
| `dflash` | Qwen3.6-35B-A3B | 15 | ✅ `z-lab/…-DFlash` | non-causal · ต้อง `--attention-backend flash_attn` |

- **DFlash drafter attention-backend แยกตามตระกูล + ต้องเซ็ต 2 ที่** (⚠️ AEON): Qwen3.x→`TRITON_ATTN`, Gemma4→`flash_attn`;
  drafter **ไม่** inherit `--attention-backend` ของ target → ต้องใส่ใน `--speculative-config` JSON ด้วย
- **n_spec เป็นปุ่ม latency↔concurrency** (⚠️ AEON): n=15 เดี่ยวเร็วสุด (~34.7 tok/s) แต่ครึ่ง KV concurrency;
  n≤4 เวลารับ concurrency สูง
- **MTP อาจไม่เสถียร** → eugr ship `-no-mtp` variant เป็น fallback ทุกโมเดล A3B ✅
- **DFlash 3 บั๊กที่ทำ acceptance ตก** (⚠️ AEON): rejected-context + prefix-caching → acceptance ไหลลง 0% ต้อง restart;
  drafter SWA ไม่ทำงาน → 0% หลัง 2048 tok; Gemma4 ขาด embed-norm/logit-softcap

## 4. Per-model serving cheat-sheet

### DeepSeek-V4-Flash (2× Spark) — ⚠️ vendor-specific (Anemll image) แต่ shape เชื่อได้
Source: `MiaAI-Lab` `docker-compose.dspark.yml:187-223`. **อย่า copy flag ดิบ — image เราต่าง**
- `--tensor-parallel-size 2 --pipeline-parallel-size 1`, `--distributed-executor-backend mp` (ไม่ใช่ Ray)
- `--kv-cache-dtype nvfp4_ds_mla` (ต้องมี flashmla fp8-kernel fix; ไม่งั้นใช้ `fp8_ds_mla`) — ✅ กฎ arch_requirements เดิมถูก
- `--block-size 256`, `--max-model-len 1048576`, `--max-num-seqs 6`, `--long-prefill-token-threshold 1024`
- `--speculative-config method=dspark num_speculative_tokens=5 draft_sample_method=probabilistic`
- `--tokenizer-mode/tool-call-parser/reasoning-parser deepseek_v4` + `--reasoning-config`
- `--moe-backend flashinfer_b12x`; env `VLLM_USE_B12X_MOE=1 CUTE_DSL_ARCH=sm_121a VLLM_USE_BREAKABLE_CUDAGRAPH=0`
  (`=0` ให้ +28.6% C1 decode), `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `NCCL_NVLS_ENABLE=0`
- perf อ้างอิง: C1 ~73-76 tok/s (65 @128K), C6 ~180-191; verify ถึง 900K single-stream
- **กับดัก concurrency**: `--max-num-seqs>1` ต้องมี build patch (request-stable slot + ragged-context) +
  `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` · 4 KV groups (1 MLA+3 SWA-MLA) → ตั้ง
  `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` ไม่งั้น shared prefix ยาว decode ออกขยะ

### Qwen3.5 / 3.6 — ✅ mapping ที่เราใช้ถูกสำหรับ INT4-AutoRound
- tool-parser: INT4-AutoRound 122B + 3.6-A3B → `qwen3_xml` ✅ (ตรงกับ arch_notes เรา);
  **exception**: 122B-**FP8** ใช้ `qwen3_coder`-parser ⚠️ (ราย checkpoint — mapping เราคงเดิม, จำ exception นี้ไว้)
- อย่าเปิด `--enable-prefix-caching` (DeltaNet hybrid → output ผิด) ✅
- Marlin/INT4 recipe ตั้ง `VLLM_MARLIN_USE_ATOMIC_ADD=1`; INT4-AutoRound ต้อง `--trust-remote-code` + chat-template ROPE fix
- 3.6-A3B-NVFP4: `--moe-backend marlin --kv-cache-dtype fp8 --attention-backend flashinfer`, gpu-util 0.4, max-num-seqs 4, mtp k=3 (มี `-no-mtp` fallback)

### Nemotron-3.x — ✅ arch class ใหม่ (hybrid Mamba/SSM)
- ต้อง `--mamba-backend flashinfer --mamba-ssm-cache-dtype {float16 Lightning|float32 Super} --mamba-cache-mode align`
  `--enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5` (eugr recipes)
- Nano = `solo_only`, cutlass MoE, มี reasoning-parser plugin แยก

### Gemma4-26B-A4B (NVFP4 CUTLASS) — ⚠️ AEON
- DFlash `attention_backend: flash_attn` (ไม่ใช่ triton), n=10-11, `max-num-seqs 128` (ปลดล็อกจาก cap 32) → 1151 tok/s @c128
- gpt-oss-120b MXFP4: ต้อง mxfp4 image + `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1` + `--mxfp4-backend CUTLASS`

## 5. Recipe schema — สิ่งที่ eugr มี แต่ LMDS candidate ยังไม่มี (พิจารณารับ)

- **`cluster_only` / `solo_only`** topology gate ต่อโมเดล (DeepSeek/MiniMax/Step = cluster; gpt-oss/Nemotron-Nano/Diffusion = solo)
  → planner ควรปฏิเสธ deploy ผิด topology พร้อมข้อความช่วยเหลือ
- **`recipe_version`** schema-version gate (เตือนเมื่อ candidate ใช้ field ที่ build นี้ไม่รู้จัก)
- **`build_args` → image variant** (`--exp-mxfp4`→mxfp4 image, `--exp-b12x`→b12x image) เป็น first-class requirement
- **`env` block เป็น serving-value payload** — publish per-model env dict ไม่ใช่แค่ CLI flags
- LMDS ใช้ Jinja อยู่แล้ว (eugr ใช้ `str.format()` และบ่นเองว่าอยากได้ Jinja) → เรานำหน้าตรงนี้

## 6. Multi-node / STACKED (2× Spark) — ✅ corroborated eugr+MiaAI

- **executor = `mp`** (multiprocessing) หรือ torchrun-style ผ่าน `vllm serve --nnodes N --node-rank R --master-addr --master-port`
  — **ไม่ต้อง Ray** (ทั้งสอง repo default no-Ray)
- **head = rank 0** (ถือ OpenAI API), **worker = rank 1 + `--headless`**; **worker start ก่อน head**
  (head mid-load ห้ามโดน SIGKILL ระหว่าง worker handshake)
- **RoCEv2 GID index ต้อง auto-resolve จาก sysfs ต่อ node** — literal ตายตัวจะ **drift หลัง reboot** ทำ NCCL init ค้าง ⚠️ MiaAI
- **dual-twin ConnectX**: NCCL ต้อง bind ทั้งสอง twin (`NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`) จึงได้ bandwidth เต็ม ⚠️ eugr
- mesh (switchless) vs switched มี NCCL env ต่างกัน; power-of-2 nodes (2/4/8) สำหรับ TP, 3-node mesh สำหรับ PP/DP
- โมเดลใหญ่ (Qwen3.5-397B, MiniMax): ใช้ `mods/drop-caches` + `--earlyoom` กัน OOM ตอน load

## 7. Benchmark methodology (เอาเข้า `lmds bench`) — ⚠️ AEON

- **6 หมวด × 4 prompt**: reasoning/math/code/prose/dialogue/summary — spread สูงถึง 2.5× (code 39.6 vs prose 16.0)
  → single-prompt bench หลอกตา ต้องแยกหมวด (สอดคล้องกับ 5-class ของเรา)
- **decode-only tok/s = 1000/TPOT** แยกจาก wall tok/s (รวม TTFT) · warmup = ทิ้ง call 16-token · round1=cold, round2=steady
- **reasoning-parser gotcha**: โมเดล reasoning stream ลง `reasoning_content` ไม่ใช่ `content` → ต้องนับทั้งคู่ +
  ส่ง `chat_template_kwargs={"enable_thinking":false}` ไม่งั้นได้ 0 หลอกๆ ทั้งที่ HTTP 200
- **liveness-gated concurrency sweep** (`validate_sweep.py`): ยิง c∈{1,4,8,12,…} เช็คทั้ง `/v1/models==200`
  **และ** container-alive แต่ละ level, break เมื่อตาย, exit 1 → เป็น crash-under-load gate ที่คมกว่าดู throughput เฉยๆ

## 8. Quant quality gate (ถ้าจะทำ `lmds verify-quant`) — ⚠️ AEON `gemma4-nvfp4/`

- A/B ผ่าน **serving path จริง** (NVFP4 โหลดใน transformers เปล่าไม่ได้) เทียบ BF16 baseline vs quantized
- 4 metric: MMLU (argmax top-logprob A/B/C/D), HumanEval (syntactic+functional), IFEval
- ผล: FP8 ~lossless (MMLU parity), NVFP4 MLP-only ตก MMLU ~3.6 pt → quantify tradeoff อย่าเดา
- **ใช้ MMLU balanced ทั้ง 57 subject** อย่าใช้ default ที่ abstract_algebra ขึ้นก่อน (worst-case ของ quant, หลอกตาแรง)
- exclude `lm_head`(tied)/embeddings/vision ไว้ BF16 เสมอ ไม่งั้น shape mismatch ตอน vLLM load

## 9. จาก albond (training repo — คนละ domain) — inference-relevant ชิ้นเดียว

- Qwen3.5 มี **MTP head ในตัวจาก HF** → เปิด speculative decoding ได้ฟรี (ย้ายเข้า arch_notes แล้ว).
  repo นี้เป็นการ **เทรน** MTP head warm-start บน fine-tuned base — LMDS เป็น deploy ไม่เกี่ยวโดยตรง
- ยืนยัน SM121 hardware facts (§1) เป็นอิสระ

---

_อัปเดตล่าสุด: 2026-08-19 · สกัดโดย review 5 community repo (albond, dgxtop, eugr, MiaAI-Lab, AEON-7)._
_`dgxtop` = monitor TUI ธรรมดา ไม่มีของเข้า LMDS._
