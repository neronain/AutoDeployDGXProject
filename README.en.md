# AutoDeployDGXProject — Local Model Deploy Studio (LMDS)

> ⚡ Built and maintained by **neronain** — [facebook.com/neronain.minidev](https://www.facebook.com/neronain.minidev)
>
> 🇹🇭 ภาษาไทย: [README.md](README.md) — the Thai README and `docs/` are the primary documentation.
> This page is a summary for English readers; the CLI itself speaks Thai.

A CLI for Ubuntu that takes a **Hugging Face model link** (repo or direct `.gguf` file) and uses an
**LLM API** (OpenAI, Gemini, MiniMax, or any OpenAI-compatible endpoint — including your own local
model) as its "brain" to analyse the model and produce a **validated deployment bundle** for:

- **NVIDIA DGX Spark** — single node or stacked (multi-node)
- **Ubuntu + RTX GPU** — ordinary local AI servers (x86_64)

> **Supported model sources today: Hugging Face only.** Ollama and NVIDIA NGC links are phase 2 —
> passing one produces a clear "not supported yet" message. The `anthropic` provider can be
> configured but its adapter is also phase 2.

## Core design principle

> **Deterministic core + LLM assist** — the LLM never writes Bash. Every script is rendered from a
> reviewed template. The LLM only researches the model and fills in a fixed JSON schema
> (`DeploymentPlan`); memory-fit and token-budget maths are pure code. Every bundle must pass
> quality gates (`bash -n`, audit rules, SHA-256) before it reaches the user.

## Quick start

```bash
git clone https://github.com/neronain/AutoDeployDGXProject
cd AutoDeployDGXProject && ./install.sh     # checks the machine, offers to configure provider + completion

lmds hardware                               # GPU / RAM / disk / Docker / target profile
lmds inspect Qwen/Qwen3-32B                 # analyse + fit check, no files written
lmds deploy https://huggingface.co/Qwen/Qwen3-32B --target dgx-spark-single
```

`deploy` walks through: analyse → plan → **confirm** (approve flags, adjust context) → render →
8 quality gates → bundle + ZIP. Then on the target machine:

```bash
cd bundles/<slug>
./<slug>-single.sh download && ./<slug>-single.sh verify-files
./<slug>-single.sh start && ./<slug>-single.sh test-text
```

Run `./<slug>-single.sh help` for full English documentation of every option, environment variable,
and how to set an API token — it is generated with that bundle's real defaults filled in.

## Managing what is running

```bash
lmds ps                  # host + every model: status, port, endpoint
lmds list                # every bundle + status + engine/port/context/features + autostart
lmds start|stop|restart <name>
lmds logs <name> -f      # live tail
lmds enable <name>       # systemd autostart after reboot
lmds repair <name>       # re-fetch missing/corrupt files, then verify
lmds remove <name>       # delete everything (--keep-weights to keep the download)
```

`lmds ps` also adopts **containers you started yourself** (vLLM / llama.cpp / Ollama / TGI) — they can
be stopped, restarted, tailed and enabled too. Stopping those uses `docker stop`, never `docker rm -f`.

Tab completion covers commands, bundle names and target presets: `lmds --install-completion`.

## Requirements

- Ubuntu 22.04 / 24.04 (ARM64 or x86_64) — the tool itself also runs on macOS for development
- Python 3.10+
- At least one LLM provider: OpenAI / Gemini / MiniMax / OpenAI-compatible (Ollama, local vLLM —
  no key needed) — or none at all with `--no-llm` (rule-based mode)
- Docker + NVIDIA Container Toolkit on the machine that will serve the model
- Free disk ≈ *(model size × 1.2) + 25 GB* — the vLLM runtime image alone is ~10–20 GB

## Security notes

The endpoint a bundle serves binds to `0.0.0.0` with **no API key by default** — anyone on the network
can use it. Use `--bind 127.0.0.1` or set `API_KEY`. Model metadata (model card, `config.json`, file
list) is sent to whichever LLM provider you configure; weights, keys and tokens never leave the
machine. Full details: [SECURITY.md](SECURITY.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/INSTALL.md](docs/INSTALL.md) | Prerequisites, disk layout, proxy/air-gapped, provider setup (incl. local AI), how models are fetched and run, smoke test |
| [docs/USAGE.md](docs/USAGE.md) | Full usage guide: deploy, controller commands + env, fleet management, target presets, troubleshooting |
| [docs/PRD.md](docs/PRD.md) | Product requirements, architecture, security, risks |
| [docs/CLI_SPEC.md](docs/CLI_SPEC.md) | CLI specification (unimplemented parts marked ❌) |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Milestones and phases |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup, invariants, how to add a preset/provider/gate |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## License

**Proprietary — all rights reserved.** See [LICENSE](LICENSE).

Bundles you generate belong to you and may be used, modified and redistributed freely. Model weights,
container images and third-party runtimes are covered by their own licenses.
