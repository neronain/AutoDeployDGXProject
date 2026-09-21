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
  — **และค่านี้ขึ้นกับจำนวน rank ด้วย ไม่ใช่แค่ตัวโมเดล: 0.80 ที่ TP4 แต่ 0.75 ที่ TP8 · ดู §6**
- **EC ล็อก GPU clock ต่ำกว่า 1 GHz ได้โดย `nvidia-smi` ไม่แสดงอะไรผิด** — อาการเดียวคือช้า
  **ต้องรัน burn check ก่อนเชื่อ benchmark ใด ๆ บนฟลีต Spark · ดู §10**
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

**gpu-util ต้องลดลงเมื่อ rank เพิ่ม** (เพิ่ม 2026-09-20) — **0.80 ที่ TP4 · 0.75 ที่ TP8**

NCCL แบบ 8 ทางจอง **~24 GiB ต่อ rank นอกงบของ vLLM** งบจริงจึงเป็น
`gpu_util × 121.7 GiB + 24 GiB + OS ต้องอยู่ใน 121.7 GiB` — ที่ TP8 ค่า 0.80 ไม่ขึ้น
ส่วน 0.77 หัวเครื่องเหลือ 1.1–1.4 GiB ตอน prefill 500K–900K จึงลงมาที่ 0.75
**นี่คือช่องว่างในโมเดลคำนวณ ไม่ใช่แค่ตัวเลขผิด** — ค่าคงที่ต่อเครื่องไม่พอ ต้องโตตามจำนวน rank

| ค่า | ที่มา | ระดับหลักฐาน |
|---|---|---|
| 0.80 ที่ TP4 | `Tech2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark` boot 10 | **หลักฐาน** — head log, `args.json`, `kv-context.txt` commit ไว้ครบ |
| 0.75 ที่ TP8 + ตัวเลข ~24 GiB/rank | `im0xMagnus/deepseek-v4.1-flash-uncensored-8x-dgx-spark` README | **คำกล่าวอ้างของผู้เขียน** — คลัสเตอร์ 8 เครื่องทั้งชุดมี result file เดียว ไม่มี log ดิบ |

> **ข้อขัดแย้งที่ยังไม่ได้ตัดสิน** — อีกรีโป 8 เครื่อง (`im0xMagnus/glm-5.3-uncensored-8x-dgx-spark`)
> ใช้ **0.80 ที่ TP8** กับ GLM-5.3 และบอกว่า 0.86 โดน OOM kill ตอน boot แรก · คนละโมเดล คนละขนาด
> weights และรีโปนั้น **ไม่มีโฟลเดอร์ `results/` เลย** · กฎที่ปลอดภัยกว่าและเข้ากับทั้งสองชุด:
> **เพดานคือ host headroom ไม่ใช่เพดานของ GPU — ยิ่ง rank เยอะ NCCL ยิ่งกินนอกงบ ต้องลด gpu-util ลง**
> ⚠️ **ยังไม่ได้วัดบนฟลีตของเราเอง**

**สถานะฝั่ง LMDS (2026-09-21): ⏸ ยังไม่ใส่พจน์ที่สเกลตาม rank — โดยตั้งใจ**

ใส่ **โครงสร้าง** ไว้ครบแล้ว (`fit/analyzer.nccl_reserve_gb_per_rank()` + `gpu_util_ceiling()`
ที่เขียนเป็นสูตร `(total − X − OS)/total` ไม่ใช่ตาราง 0.75/0.80) แต่ **ค่าเริ่มต้นให้ผลเท่า
พฤติกรรมเดิมทุกจำนวน rank** — สวิตช์ `NCCL_COMM_BUFFER_GB_PER_PEER = None` ยังปิดอยู่

ทำไมถึงยังไม่เปิด — สามข้อ:

1. หมุด "~24 GiB/rank ที่ TP8" เป็น **คำกล่าวอ้างของผู้เขียน** และผู้เขียนคนเดียวกันขัดกับตัวเอง
   (0.80 ที่ TP8 กับ GLM-5.3 · รีโปนั้นไม่มี `results/`)
2. **~24 GiB/rank อยู่ใกล้ ~27 GiB/rank ของ COW break** (fused-MoE expert copy ·
   `docs/UPGRADE-2026-09.md` §1.7 R8) มาก — กลไกนั้นโตตาม **checkpoint** ไม่ใช่ตาม rank
   และ **แยกออกได้ที่ N=1 ซึ่งไม่มี NCCL อยู่ในโปรเซสเลย** (เครื่องเดียว ไม่ต้องมีสาย)
   → **ข้ออ้างว่าก้อนนี้เป็นของ NCCL ยังไม่ถูกพิสูจน์ และล้มได้ง่ายกว่าที่คิด**
3. **ฟลีตเรารัน TP4/TP8 ไม่ได้** — `docs/HCA-DUAL-TEST.md` §3 (recon ของเราเอง 2026-09-20):
   ConnectX-7 แค่ 5 ใบใน 11 เครื่อง ต่อถึงกัน 2 คู่ที่เป็นเกาะคนละเกาะ →
   **world size สูงสุดบน fabric เดียว = 2**

การวัดที่จะตัดสินเรื่องนี้อยู่ที่ `docs/GPU-UTIL-RANK-PROTOCOL.md` (เกณฑ์เขียนไว้ก่อนรัน) ·
§5 สั่งไว้ล่วงหน้าว่าห้ามใส่พจน์ rank จนกว่า verdict = R เพราะถ้าผลเป็น C หรือ H
**ทุก stacked fit จะผิดในแบบที่ดูน่าเชื่อถือกว่าเดิม**

| ranks | ที่ `fit` ใช้วันนี้ | เพดาน gpu-util ที่ได้ | หมายเหตุ |
|---|---|---|---|
| 1 | 0 | 0.90 | ไม่มี NCCL ในโปรเซส |
| 2 | 3.0 GB | 0.85 | **วัดเองบน 2×Spark** · = ค่าที่ recipe stacked ใน catalog ใช้อยู่ |
| 3+ | 3.0 GB (คงที่) | 0.85 | **อ้างนอกช่วงที่วัด** — `fit` ติดป้ายเตือนไว้ในโน้ตแล้วว่าอาจต่ำกว่าจริง |

⚠️ ตาราง 0.80@TP4 / 0.75@TP8 ข้างบน **ยังไม่ถูกเข้ารหัสลงในสูตร** — เราไม่แกล้งรู้สิ่งที่วัดไม่ได้

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

## 10. GPU clock latch บน GB10 — ✅ artifact-backed (Tech2wild)

**อ่านหัวข้อนี้ก่อน benchmark ทุกครั้ง** — เป็นข้อเดียวในไฟล์ที่ทำให้ตัวเลขทุกตัวข้างบนโกหกได้

**อาการ** — EC (embedded controller) ล็อก GPU ไว้ที่ **631–949 MHz** ทั้งตอน idle และตอนมีโหลด
ขณะที่เครื่องอื่นในชุดเดียวกันวิ่ง 2177–2561 MHz · `nvidia-smi` **ไม่แสดงอะไรผิดเลย**:
P0 · persistence on · application clocks 2418 (max 3003) · ไม่มี clock event reason
(ไม่ power cap ไม่ thermal ไม่ HW slowdown) · ไม่มี locked clock · kernel log สะอาด

**ทำไมมันร้ายใน TP** — ทุก collective รอ rank ที่ช้าที่สุด เครื่องที่ latch สองเครื่องจึงลากทั้งคลัสเตอร์
ในชุด 4 เครื่องที่เจอ: boot ก่อนแก้ count 41.5 / code 32.9 tok/s · boot หลัง power-cycle
count 60.8 / code 57.1 · **ระวัง**: สอง boot นั้นเปลี่ยน gmu 0.78 → 0.80 และเปิด tools/vision ด้วย
เจ้าของรีโปยกให้เป็นผลของ clock latch — ไม่ใช่การทดลองที่คุมตัวแปรตัวเดียว

**สาเหตุ** — EC ตัดสินใจ DVFS อยู่ใต้ OS และ latch ค้างในสถานะคล็อกต่ำได้ ·
**รีบูตปกติหรือ soft shutdown ไม่หาย** เพราะ EC ยังกินไฟ standby ตราบที่ adapter เสียบอยู่

**วิธีแก้ทางเดียว**

1. ปิดเครื่อง
2. **ถอดปลั๊ก adapter 30–60 วินาที** (นานกว่านั้นได้)
3. เสียบกลับแล้วเปิด

ระหว่างถอด เช็คด้วยว่าเป็น adapter ตัวเดิมและเสียบแน่น — EC ลดคล็อกเองเมื่อเห็นว่าไฟไม่พอ

**วิธีตรวจ — burn check 15 วินาที** fp16 matmul 4096² แล้วอ่าน `clocks.sm` + `power.draw`
ที่วินาทีที่ 12

```
ปกติ     ≥80 W · 2.2–2.4 GHz · 75–90 TFLOPS
latched  ~700–950 MHz · <20 W
เกณฑ์ fail  <50 TFLOPS
```

ค่าที่วัดได้จริงหลัง power-cycle ในรีโปต้นทาง: 84.0 / 76.6 TFLOPS บนสองเครื่องที่ latch
เทียบกับ 89.2 / 88.8 บนสองเครื่องที่ไม่ latch — **หลัง power-cycle ทั้งสี่เครื่องเท่ากัน**

ที่มา: `Tech2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark` → `docs/gpu-clock-latch.md` (2026-09-10)
พร้อมตารางค่าต่อเครื่องและสคริปต์ `tools/recover.sh` ที่ commit ไว้ — **artifact-backed**

**สถานะใน LMDS (2026-09-21)**: มีแล้ว — `lmds burn` / `lmds burn --all` (`src/lmds/hardware/burn.py`) ·
`lmds bench run` เรียกให้เองก่อนวัดทุกครั้งแล้ว **เก็บผลไปกับตัวเลขที่วัดได้** (`environment.clocks`)
จึงรู้ทีหลังได้ว่ารอบไหนวัดตอนคล็อกจริง · `lmds cluster doctor` ที่ผ่านทุกข้อชี้ไปที่ `lmds burn --all`
เพราะหมอตัวนั้นอ่านอย่างเดียวจึงมองข้อนี้ไม่เห็น

### วัดเองบนฟลีตของเรา — 11 GB10 (2026-09-21) · ✅ artifact-backed

รันจริงด้วย `lmds burn --node <ชื่อ> --json` ทีละเครื่อง · fp16 matmul 4096² 15 วินาที
อ่านที่วินาทีที่ 12 · **ไม่เจอเครื่องที่ latch เลยสักเครื่อง**

| ช่วงที่วัดได้ | ต่ำสุด | สูงสุด |
|---|---|---|
| TFLOPS | **80.4** (msi-6) | **90.7** (dgx-spark04) |
| คล็อก SM | **2119 MHz** (dgx-veerasiam) | **2359 MHz** (spark-worker) |
| ไฟ | **86.1 W** (dgx-spark03) | **95.5 W** (msi-6) |
| อุณหภูมิ | 63 °C (spark-head) | 81 °C (msi-1) |

ทั้ง 11 เครื่องได้ `kind: ok` · `throttle: []` ทุกตัว · ไม่มีตัวไหนเข้าใกล้ลายเซ็นของ latch
(631–949 MHz · <20 W) แม้แต่น้อย — ตัวที่คล็อกต่ำสุดยังสูงกว่าเพดานของ latch **2.2 เท่า**

**สิ่งที่ตัวเลขชุดนี้ยืนยันกับหัวข้อนี้**: ช่วงปกติที่เขียนไว้ (≥80 W · 2.2–2.4 GHz ·
75–90 TFLOPS) ตรงกับของจริงทั้งสามมิติ · และค่า 89.2 / 88.8 ที่รีโปต้นทางวัดได้บนเครื่อง
ที่ไม่ latch อยู่กลางช่วงของเราพอดี

**สิ่งที่ยังไม่ยืนยัน**: เราไม่เคยเห็นเครื่องที่ latch ด้วยตาตัวเอง — เกณฑ์ตก (<50 TFLOPS)
กับเส้นอธิบาย (1500 MHz / 50 W) จึงยังพิสูจน์จากฝั่งเรา*ไม่ได้* · เจอเมื่อไรต้องบันทึกทันที

> **ข้อสังเกตสองอย่างที่ตัวเลขชุดนี้เปิดให้เห็น** (ไม่ใช่ความผิดปกติ แต่ควรรู้)
> · เครื่องตระกูล `msi-*` ร้อนกว่าตระกูล `spark-*` อย่างสม่ำเสมอ (75–81 °C เทียบกับ 63–72 °C)
> ทั้งที่ไม่มีตัวไหน throttle — เป็นเรื่องของตัวถัง/ลม ไม่ใช่ของ GPU
> · `msi-6` กินไฟมากที่สุด (95.5 W) แต่ได้ TFLOPS ต่ำที่สุด (80.4) ขณะที่คล็อกสูงเป็นอันดับสอง
> — ยังอยู่ในช่วงปกติ แต่เป็นเครื่องที่ควรดูซ้ำก่อนเอาไปเทียบ benchmark กับเครื่องอื่น

ข้อมูลดิบต่อเครื่องเก็บไว้นอก repo (มีชื่อเครื่องกับ IP) — ดู `docs/SMOKE-TEST-2026-09.md`

> **เกณฑ์ตัดสินยังไม่เคยเจอเครื่องเสียจริง** — `<50 TFLOPS` มาจากหัวข้อนี้ตรง ๆ ส่วนเส้นแบ่ง
> คล็อก 1500 MHz / ไฟ 50 W ที่ใช้*อธิบาย*ว่าทำไมถึงตก เป็นจุดกึ่งกลางที่เลือกเอง ไม่ได้วัดมา

## 11. อย่าเชื่อเลข "KV pool tokens" ที่ engine พิมพ์ — ✅ artifact-backed (Tech2wild)

vLLM พิมพ์ `max_concurrency × max_model_len` (`kv_cache_utils.py:2306`) **ไม่ใช่ความจุจริง**
เลขนั้นจึงกระโดดตาม `--max-model-len` ที่ตั้ง ทั้งที่หน่วยความจำเท่าเดิม

วัดจริงบนเครื่องเดียวกัน คนละ `max-model-len`:

| max-model-len | KV cache ที่ได้จริง |
|---|---|
| 500K | **29.97 GiB** |
| 300K | **30.14 GiB** |

**เกือบเท่ากัน** ส่วนตัวเลข token ต่างกันลิบ · ที่มา: `runs/2026-09-19-speedrun2/README.md`
พร้อม `results/s2-00-baseline/kv-context.txt` และ `results/e00-ctx300k/kv-context.txt`
ที่ commit ไว้ทั้งคู่ — **artifact-backed**

**ให้อ่าน GiB จากบรรทัด `Available KV cache memory:` เสมอ** ไม่ใช่จำนวน token

**สถานะฝั่ง LMDS (2026-09-21) — ตรวจแล้ว และของเดิมไม่ได้ผิดอย่างที่เขียนไว้**

ยืนยันข้อสังเกตต้นทางซ้ำด้วย log ของเราเอง (ไม่ใช่แค่ `kv_cache_utils.py` ของ vLLM):
Nemotron `4.50 × 262,144 = 1,179,648` **ตรงเป๊ะ** · Qwopus `508,031 ÷ 262,144 = 1.9380` → พิมพ์ `1.94`

แต่ `fit/sizing.py` **อ่าน GiB อยู่แล้ว** (`kv_pool_gib` มาจากบรรทัด `reserved … GiB` /
`Available KV cache memory:`) ส่วน token เป็นแค่ *ตัวหาร* · และที่ context เดิม
`pool ÷ (conc × ctx) × ctx ≡ pool ÷ conc` — max_model_len ตัดกันพอดี ตัวเลข 1.33/6.19 GiB ต่อคำขอ
ที่ pin ของเรายืนอยู่บนนั้นจึงถูกมาตลอด

บั๊กจริงแคบกว่า: **ค่าที่วัดได้ผูกกับ context ที่วัด** แล้วถูกเอาไปคูณที่ context อื่นตรง ๆ ·
โมเดล hybrid มี state ที่ไม่โตตาม context ทำให้ `KV(ctx) = a × ctx + b` ไม่ใช่สัดส่วนตรง —
Nemotron ที่ `--context 65536` เดิมตอบ 0.33 GiB ต่อคำขอ ของจริง 0.58 (**ต่ำไป 75%**)
ตอนนี้เก็บ `kv_measured_context` ไว้เสมอแล้วถอด `b` ออกก่อนคิดใหม่ · ดู `docs/UPGRADE-2026-09.md` §1.6

---

## ความสดของข้อมูล — อ่านก่อนใช้ตัวเลขในไฟล์นี้

| ส่วน | เก็บข้อมูลเมื่อ | แหล่ง |
|---|---|---|
| §1–§9 | **2026-08-19** | review 5 community repo (albond, dgxtop, eugr, MiaAI-Lab, AEON-7) — ไม่ได้ตรวจซ้ำตั้งแต่นั้น |
| §1 (gpu-util ตาม rank) · §6 · §10 · §11 | **2026-09-20** | รีโปไต้หวัน 3 ชุด · run จริง 2026-09-10..19 |
| §10 (หัวข้อย่อย "วัดเองบนฟลีตของเรา") | **2026-09-21** | **ของเราเอง** — `lmds burn` บน GB10 11 เครื่อง |

`dgxtop` = monitor TUI ธรรมดา ไม่มีของเข้า LMDS

**กฎ**: ที่ใดที่ §1–§9 ขัดกับ run จริงของรีโปไต้หวัน ให้ถือว่า run จริงถูกแล้วไปตรวจซ้ำ ·
ทุกข้อที่มาจากรีโปภายนอกบอกไว้ว่าเป็น **artifact-backed** (มี log/JSON commit ไว้) หรือ
**คำกล่าวอ้างของผู้เขียน** (มีแต่ README)

_อัปเดตล่าสุด: 2026-09-21 (เพิ่มสถานะฝั่ง LMDS ใน §6 และ §11 — ตัวหลักฐานภาคสนามไม่ถูกแก้)_
