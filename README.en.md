<div align="center">

# LMDS · Local Model Deploy Studio

**From a Hugging Face link to a server that actually answers — on your own machine**

Deploy language models to **NVIDIA DGX Spark** and **Ubuntu + RTX**, one machine or
several acting as one. Nothing leaves the machine except what you ask for.

[![version](https://img.shields.io/badge/version-0.5.1-1f5fbf)](CHANGELOG.md)
[![tests](https://img.shields.io/badge/tests-1273-17703f)](tests/)
[![platform](https://img.shields.io/badge/platform-Ubuntu%2022.04%20%7C%2024.04-555)](docs/INSTALL.md)
[![arch](https://img.shields.io/badge/arch-ARM64%20%C2%B7%20x86__64-555)](docs/INSTALL.md)
[![python](https://img.shields.io/badge/python-3.10%2B-3776ab)](pyproject.toml)
[![license](https://img.shields.io/badge/license-proprietary-8a5300)](LICENSE)

**[Install](docs/INSTALL.md)** · **[Usage](docs/USAGE.md)** · **[Multi-node](docs/RUNBOOK-MULTI-NODE.md)** · **[ภาษาไทย](README.md)**

Built and maintained by **neronain** — [facebook.com/neronain.minidev](https://www.facebook.com/neronain.minidev)

</div>

> 🇹🇭 The Thai [README.md](README.md) and `docs/` are the primary documentation and the CLI speaks
> Thai. This page is the summary for English readers; the **web console is in English**.

---

## What it looks like

<div align="center">

<img src="docs/img/fleet.png" alt="LMDS web console — the whole fleet on one page" width="900">

*Every machine on one page — GPUs, RAM, temperature, how many models are serving. A machine
with no GPU knows it is a control plane and refuses to pull weights down.*

<img src="docs/img/model-scores.png" alt="Model scores measured against the real server" width="900">

*Scores measured through the running server own OpenAI API — decode tok/s, TTFT, the context
it will actually accept, and seven capabilities probed one at a time. Comparable across engines
and across machines.*

</div>

## The problem it solves

Getting a model onto your own hardware is rarely hard because you cannot find the command. It is
hard because **commands that look entirely correct return wrong results with no error**: a context
silently cut to a tenth of what the machine could hold, tool calling switched on that never
actually converts a reply, a 200G link that negotiated down to 50G, a KV cache estimated twenty
times too large so the context is capped for no reason.

LMDS is what came back from running all of that for real and turning each symptom into a check.

| | |
|---|---|
| 🧮 **Arithmetic is code, not the LLM** | Memory fit, KV cache, token budgets, link speed. The LLM only researches the model and fills a fixed JSON schema — it **never writes Bash**. |
| 🛡️ **Every bundle passes gates first** | `bash -n`, audit rules, SHA-256 checksums. No pass, no ZIP. |
| 🔍 **Told while it is still fixable** | Not when users complain about latency. Every check comes from something that actually broke on real hardware. |
| 🔌 **Works with no LLM at all** | Rule-based mode uses recipes proven on real machines. Air-gapped is fine. |

## Three commands

```bash
./install.sh                 # installs missing Docker / NVIDIA toolkit too, asking before each sudo
lmds hardware                # what this machine is, and what target profile it maps to
lmds deploy Qwen/Qwen3-32B   # analyse → plan → confirm → bundle + ZIP that passed every gate
```

Then, on the target machine:

```bash
cd bundles/<slug>
./<slug>-single.sh download && ./<slug>-single.sh verify-files
./<slug>-single.sh start && ./<slug>-single.sh test-text
```

---

## Three questions few tools answer

### 1 · "At this context, how many people can use it at once?"

Most tools report the largest context that fits — which is, by definition, the one where a single
conversation fills the KV pool exactly. Set it and the second request queues, with nothing saying so.

```
KV bf16 · 120 KiB per token
  context      KV each     at once
   32,768       3.8 GB        14.1
  131,072        15 GB         3.5
  262,144        30 GB         1.8   ← the value you typed
```
> • Fits, but serves 1.8 conversations at once — one full-length chat takes almost the whole pool
> • Switching the KV cache to fp8 halves it: 30 GB → 15 GB, 1.8 concurrent → 3.5
> • 2 nodes — this budget does not yet include NCCL buffers across the network

Shown in the CLI **and in the web console while the number is still being typed**. Handles GQA and
**MLA** (DeepSeek-V2/V3, Kimi K2/K3), which stores one latent instead of a key and a value — one
formula does not fit both families.

### 2 · Several machines, one model

> **Stacked does not mean faster — it means too big for one machine.**
> Anything that fits on one machine runs faster on one machine.

| | Single | Stacked |
|---|---|---|
| Engine | vLLM **or** llama.cpp | **vLLM only** |
| Artifact | safetensors or GGUF | **safetensors only** |
| Fast link | not needed | **required**, ≥25G (200G RoCE in practice) |
| Machines | 1 | ≤3 direct-cabled · ≤4 through a switch |

LMDS detects ConnectX/RDMA itself, says which machines can stack together, writes `cluster.env`,
and **warns when a link negotiated below what the card can do** — NVIDIA validates Spark links at
≥184 Gbit/s, and a switch port left on auto-negotiate commonly lands at 50G while everything still
looks healthy.

→ [RUNBOOK-MULTI-NODE.md](docs/RUNBOOK-MULTI-NODE.md) · [FLEET-MULTI-NODE.md](docs/FLEET-MULTI-NODE.md) · [compared with NVIDIA's own docs](docs/NVIDIA-CLUSTER-SOURCES.md)

### 3 · A web console that matches the CLI

```bash
lmds web --bind 0.0.0.0 -b      # asks for a token once, then remembers it — the link is bookmarkable
```

Deploy wizard, download + verify, start/stop/restart, port/context/slots/API key/bind, doctor,
logs, the test suite (`test-text` `test-vision` `test-tools` `bench` `stress`), `parsers`,
autostart, stacked
commands, repair, remove — **and models on other machines are controlled exactly like local ones**.

- **Readable before it is legible** — the same CPU / Unified·RAM / VRAM / Disk gauges for every
  machine, with warning colours before things run out. Values a card does not report are hidden,
  not shown as 0.
- **Machines that can stack together share a coloured frame** labelled `CLUSTER A/B`
- **Buttons appear only for commands that controller really supports**, read from its own dispatch table
- **Four text sizes** (S/M/L/XL) and light/dark/system themes, remembered per browser
- **Fetches nothing from the internet** — fine behind a proxy or fully air-gapped

> 🔒 The console can start, stop and delete models, so it binds `127.0.0.1` by default. **The
> printed link carries no token**, because URLs end up in browser history, proxy logs and referrers.
> Repeated wrong guesses from one IP back off exponentially.

**The assistant** answers from *this fleet's actual state* rather than general knowledge — "which
node is unreachable", "why won't msi-6 start". It uses the same LLM that plans deployments, hides
itself when no provider is configured, and knows the context/KV rules but is **forbidden from doing
the arithmetic**: an LLM multiplying in its head is wrong in a way that reads as authoritative.

It also **goes and looks at the machine before answering**. Ask why a model will not start and the
system opens that controller's logs, checks the GPU, disk and ports, or runs `lmds doctor` on the
target node over SSH first, then answers from what came back — the "ดูมาแล้ว: …" line above each
reply lists exactly what it checked.

Once the cause is clear enough to propose a fix, it **asks you instead of acting**:

| Choice | What happens |
|---|---|
| **แก้เลย** (do it) | Runs every step in one go |
| **ทีละขั้น** (step by step) | Runs one step, stops so you can read the result, waits for the next press |
| **ยังไม่ทำ** (hold) | Shows the commands and touches nothing |

**The LLM cannot issue commands.** It only picks entries from a fixed catalogue
(`lmds/assistant/catalog.py`); the shell command is assembled in code, and the approval ticket is
minted by the server — the only way anything runs is a human pressing a button. See
[SECURITY.md](SECURITY.md).

---

## One machine, whole fleet

```bash
lmds node add 192.168.10.21 --user ops --install   # password once → installs key + LMDS for you
lmds ps --all                     # every machine's models in one table
lmds node cluster                 # who has 200G, and which pairs can stack
lmds scan --all                   # weights already on disk anywhere — no re-downloading
lmds node push spark2 <slug>      # send the bundle you approved to another machine
lmds node clone <slug> --from msi-1 --to msi-2   # copy a model machine-to-machine, no re-download
```

<details>
<summary>All model-management commands</summary>

```bash
lmds ps                  # who is running: name, model, engine, port, ● running / ◐ loading / ○ stopped
lmds list                # every bundle + engine/port/context/features + autostart
lmds smoke <name>        # prove it runs: download → verify → start → test-text → stop
lmds start/stop/restart <name>
lmds logs <name> -f      # -n 500 = tail
lmds enable <name>       # come back after reboot (systemd) · disable = undo
lmds doctor <name>       # why download/start still fails + the command that fixes it
lmds repair <name>       # re-download missing/corrupt files, then re-verify
lmds rebuild <name>      # regenerate the same bundle with the current logic
lmds set <name> --image <digest> --tool-parser qwen3_xml --extra-args "…"   # one value for every way start is called
lmds adopt               # bring a model that was running before LMDS into the system
lmds remove <name>       # delete everything (--keep-weights keeps the weights)
lmds recipes --sync / --publish <name> --features tools,vision
```

`lmds ps` also shows **containers not deployed through LMDS** (vLLM / llama.cpp / Ollama / TGI
already running) — stop/restart/logs/enable work the same; that group uses `docker stop` and never
removes the container.

</details>

Target machines run **no daemon**, need no port beyond 22, and **no root** — membership of the
`docker` group is enough. The password is discarded the moment the key is installed; the registry
has no password field by design.

## What is supported

| | ARM64 / unified (Spark) | x86_64 / discrete (RTX) |
|---|---|---|
| **llama.cpp** | ✅ native build | ✅ docker (+ multimodal) |
| **vLLM** | ✅ docker | ✅ docker |

Hardware-validated across all five model families — GGUF, NVFP4, MoE, dense safetensors, gated
repos · **22 target presets** (7 verified on real hardware) · **903 tests**

> **Model source: Hugging Face only.** Ollama registry and NVIDIA NGC are phase 2 — passing such a
> link produces a clear "not supported yet" message with an alternative.

## Recipes — learn once, reuse across the fleet

A machine with no LLM API key deploys rule-based, which only knows "GGUF → llama.cpp"; it does not
know per-model facts (the parser, the image with the right kernels, the mmproj file), so a deploy
can succeed and the server still fail to start. **Recipes** close that gap: controllers that
**ran on real hardware** live in a central Git repository.

- **pull** — `lmds recipes --sync` fetches the latest recipes from the canonical store; `deploy --no-llm` uses them instead of guessing
- **push** — `lmds recipes --publish <slug> --features tools,vision` sends a controller you tested to the candidates store for review

Two tiers: **canonical** ([`dgx-spark-all-controllers`](https://github.com/neronain/dgx-spark-all-controllers)),
curated and pulled by every machine, and **candidates** ([`script-update`](https://github.com/neronain/script-update)),
freshly published and awaiting review. The publish target is configured (`recipes.publish_repo`);
empty means a local store, so customer fleets never touch our repositories.

Only **model values** travel (engine, image, parser, mmproj, measured capabilities). **Machine
values** (port, context, slots, gpu-util) stay in `bundle.env`, and the receiving machine re-fits
them to its own memory. Since 0.5.1 the values you set with `lmds set` (a proven image digest,
`--tool-parser`, `--reasoning-parser`, `--engine-env`, `--extra-args`) are folded into the
published header, so the store holds the recipe that actually started, not the plan's first guess.

## Updating

```bash
cd ~/AutoDeployDGXProject && git pull && ./install.sh
```

> ⚠️ **`git pull` alone is not enough.** LMDS is installed as a copy into its venv (not editable),
> so the `lmds` command keeps running the old code until `./install.sh` is run again. Existing
> config and keys are kept.

## Pairs with LiteGate (optional)

**[LiteGate · AiGatewayLocal](https://github.com/neronain/AiGatewayLocal)** is the other half:
LMDS *deploys* models onto your machines; LiteGate is the *single door* in front of all of them —
API keys, quotas, per-person permissions, and measuring what a running server **can actually do**.

Neither needs the other. Install LMDS alone to deploy and run models; LiteGate alone to put a door
in front of servers you already run; both to have LMDS build, LiteGate measure, and each tell the
other what to fix.

## Documentation

| | |
|---|---|
| [INSTALL.md](docs/INSTALL.md) | Step-by-step install — prerequisites, disk, proxy/air-gapped, providers, uninstall |
| [USAGE.md](docs/USAGE.md) | Full guide — deploy, every controller command and env var, fleet, web console, troubleshooting |
| [PREFLIGHT.md](docs/PREFLIGHT.md) | What is checked before deploying and why — every item from something that really broke |
| [NETWORK.md](docs/NETWORK.md) | Every port and protocol the system uses, who talks to whom, and what to open when forwarding ports or sitting behind a reverse proxy |
| [RUNBOOK-MULTI-NODE.md](docs/RUNBOOK-MULTI-NODE.md) | The multi-node command sequence as actually run, with real figures and timings |
| [FLEET-MULTI-NODE.md](docs/FLEET-MULTI-NODE.md) | Running many machines from one |
| [NVIDIA-CLUSTER-SOURCES.md](docs/NVIDIA-CLUSTER-SOURCES.md) | NVIDIA's clustering docs — what they confirm, what they add |
| [PRD.md](docs/PRD.md) · [CLI_SPEC.md](docs/CLI_SPEC.md) · [ROADMAP.md](docs/ROADMAP.md) | Requirements, command spec, roadmap |
| [SECURITY.md](SECURITY.md) · [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) | What leaves the machine · dev setup and rules · history |

## Requirements

- **Ubuntu 22.04 / 24.04** (ARM64 or x86_64) — development works on macOS
- **Python 3.10+**
- **Docker + NVIDIA Container Toolkit** on target machines (`./install.sh` can install both)
- **Free disk** ≈ *(model size × 1.2) + 25 GB* — the vLLM runtime image alone is 10–20 GB
- **An LLM provider** (optional): OpenAI / Gemini / MiniMax / OpenAI-compatible — or none, with `--no-llm`

The one thing `install.sh` will not do is install the **NVIDIA driver**: it needs a reboot, and on
some machines a working driver is already present while `ubuntu-drivers install` breaks on
dependencies.

## For developers

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]' && pytest
```

Rules that must not be broken, and how to add a target preset, provider or quality gate:
[CONTRIBUTING.md](CONTRIBUTING.md)

## License

**Proprietary — all rights reserved.** See [LICENSE](LICENSE).

Source being readable here grants no right to use or redistribute it. **Bundles you generate are
yours** — use, modify and pass them on freely. Third-party models, images and runtimes remain under
their own licences.

<div align="center">
<br>

Controller standard inherited from [dgx-spark-all-controllers v3.0.0](https://github.com/neronain/dgx-spark-all-controllers)

**neronain** · [facebook.com/neronain.minidev](https://www.facebook.com/neronain.minidev)

</div>
