# AutoDeployDGXProject — Local Model Deploy Studio (LMDS)

> ⚡ Built and maintained by **neronain** — [facebook.com/neronain.minidev](https://www.facebook.com/neronain.minidev)
>
> 🇹🇭 ภาษาไทย: [README.md](README.md) — the Thai README and `docs/` are the primary documentation.
> This page is a summary for English readers; the CLI itself speaks Thai.

A CLI for Ubuntu that takes a **Hugging Face model link** (repo or direct `.gguf` file) or an
**Ollama registry link**, and uses an
**LLM API** (OpenAI, Gemini, MiniMax, or any OpenAI-compatible endpoint — including your own local
model) as its "brain" to analyse the model and produce a **validated deployment bundle** for:

- **NVIDIA DGX Spark** — single node or stacked (multi-node)
- **Ubuntu + RTX GPU** — ordinary local AI servers (x86_64)

> **Supported model sources today: Hugging Face and the Ollama registry.** An
> `ollama.com/<namespace>/<model>:<tag>` link is pinned to its model-blob digest and rendered as a
> llama.cpp bundle. Ollama Modelfile/controller output and NVIDIA NGC remain phase 2. The
> `anthropic` provider can be configured but its adapter is also phase 2.

## Core design principle

> **Deterministic core + LLM assist** — the LLM never writes Bash. Every script is rendered from a
> reviewed template. The LLM only researches the model and fills in a fixed JSON schema
> (`DeploymentPlan`); memory-fit and token-budget maths are pure code. Every bundle must pass
> quality gates (`bash -n`, audit rules, SHA-256) before it reaches the user.

## Quick start

```bash
git clone https://github.com/neronain/AutoDeployDGXProject
cd AutoDeployDGXProject && ./install.sh     # installs whatever is missing, then configures provider
source ~/.bashrc                            # the installer prints exactly what to run at the end

lmds hardware                               # GPU / RAM / disk / Docker / target profile
lmds inspect Qwen/Qwen3-32B                 # analyse + fit check, no files written
lmds deploy https://huggingface.co/Qwen/Qwen3-32B --target dgx-spark-single
```

`deploy` walks through: analyse → plan → **confirm** (approve flags, adjust context) → render →
10 quality gates → bundle + ZIP. Then on the target machine:

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
lmds start|stop|restart <name>          # extra flags (--port, --gpu-util) pass through to the controller
lmds logs <name> -f      # live tail
lmds enable <name>       # systemd autostart after reboot
lmds doctor <name>       # why won't it download/start? checks + exact fix commands
lmds repair <name>       # re-fetch missing/corrupt files, then verify
lmds remove <name>       # delete everything (--keep-weights to keep the download)
```

`lmds ps` also adopts **containers you started yourself** (vLLM / llama.cpp / Ollama / TGI) — they can
be stopped, restarted, tailed and enabled too. Stopping those uses `docker stop`, never `docker rm -f`.

## One machine, or several?

**Stacking is not about speed — it is about a model not fitting in one machine.** Anything that
fits on one machine runs *faster* there, because nothing crosses the wire per token.

| | Single machine | Stacked (several machines, one model) |
|---|---|---|
| Engine | vLLM **or** llama.cpp | **vLLM only** |
| Artifact | safetensors or **GGUF** | **safetensors only** — GGUF cannot be stacked |
| Fast fabric | not needed | **required**, ≥25G (in practice 200G RoCE) |
| Target | `dgx-spark-single`, `rtx-5090`, … | `dgx-spark-stacked`, `dgx-spark-stacked-4` |

Full command sequence: **[docs/RUNBOOK-MULTI-NODE.md](docs/RUNBOOK-MULTI-NODE.md)**

## Controlling several machines from one

A site with more than one machine does not need one SSH session per box. Register a machine once
with its ip/user/password — LMDS installs its own SSH key and discards the password immediately
(the registry has no password field, on purpose).

```bash
lmds node add 192.168.10.21 --user ops --install   # password once → installs the key and LMDS
lmds node list --check                   # which machines still answer
lmds ps --all                            # every model on every machine, one table
lmds node run spark2 doctor my-model     # run any lmds command on that machine
lmds node cluster                        # who has ConnectX/200G, and which pairs can be stacked
lmds scan --all                          # models already on each machine, wherever they were put
lmds node ctl spark1 <slug> start        # run a controller step on that machine
lmds prune                               # clear registrations pointing at bundles that are gone
lmds recipes                             # configurations proven on hardware — used when no LLM key is set
```

Nodes run **no daemon** and need no port open beyond 22 — the hub calls `lmds agent info` over SSH.
That does mean **every machine needs LMDS installed on it** — the "agent" is the `lmds` command
itself, not a resident process. The hub can do that for you: `lmds node install <name>` clones and
runs `install.sh` there (skipping the sudo/Docker step, which no one can answer over SSH).
Root is not required; a user in the `docker` group is enough. Each machine reports live CPU, RAM
(or unified memory), VRAM, disk, link speed and **how many models are running** — llama.cpp can
serve several at once.

For stacking, LMDS reads `/sys` to find ConnectX/RDMA links and their speed, then groups machines
that can actually be stacked together (same GPU model and count, fast enough link on both). It
suggests the cluster IP it found but never assumes it — set it explicitly, then write it into the
bundle so the controller stops asking:

```bash
lmds node set spark2 --cluster-ip 10.10.0.2
lmds node cluster --write my-70b-model    # writes cluster.env (MASTER_IP/WORKER_IP/NCCL_SOCKET_IFNAME)
```

Details: [docs/FLEET-MULTI-NODE.md](docs/FLEET-MULTI-NODE.md)

## Web UI (optional)

```bash
lmds web                              # http://127.0.0.1:8600 — this machine only
lmds web --bind 0.0.0.0               # reachable on the LAN; a token is generated for you
```

```bash
lmds web --background                 # run detached; the terminal stays free for the CLI
lmds web --status                     # forgot the link? ask the running server
lmds web --restart -b                 # restart with a fresh token
lmds web --stop
```

One English-language page covering the whole workflow: host status (CPU, memory, VRAM, disk,
running-model count), other machines with their live resources, cluster fabric, deploy wizard, download
(which verifies afterwards), start/stop/restart, per-model port/context/slots/API key, the test
commands (`test-text`, `test-vision`, `bench`, `stress`, …), autostart, stacked commands, repair
and remove. Buttons follow what each controller actually supports, read from the script itself —
an older bundle simply won't show a command it doesn't have.

It can start, stop and delete models, so it binds to localhost by default and generates a token
whenever you expose it. The page loads nothing from the internet — it works behind a proxy or
fully air-gapped. Still CLI-only: `lmds config` and `lmds hardware`.

Tab completion covers commands, bundle names and target presets: `lmds --install-completion`.

## Requirements

- Ubuntu 22.04 / 24.04 (ARM64 or x86_64) — the tool itself also runs on macOS for development
- Python 3.10+
- At least one LLM provider: OpenAI / Gemini / MiniMax / OpenAI-compatible (Ollama, local vLLM —
  no key needed) — or none at all with `--no-llm` (rule-based mode)
- Docker + NVIDIA Container Toolkit on the machine that will serve the model
- Free disk ≈ *(model size × 1.2) + 25 GB* — the vLLM runtime image alone is ~10–20 GB

`install.sh` sets up the Docker prerequisites for you. It asks before every `sudo` step and prints the
exact command first: Docker itself, adding your user to the `docker` group, the NVIDIA Container
Toolkit (all five steps), and `python3-venv` — then verifies Docker really sees the GPU. Answer `n` to
skip any of them and it tells you the command to run yourself instead.

```bash
sudo -v && LMDS_ASSUME_YES=1 ./install.sh    # unattended: accept every prompt
LMDS_SKIP_PREREQ=1 ./install.sh              # install LMDS only, never touch Docker
```

The NVIDIA **driver** is the one thing it will not install — that needs a reboot, and on machines with
a working driver `ubuntu-drivers install` can break package dependencies. When not attached to a real
terminal (CI, piped input) the installer changes nothing on the machine.

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
| [docs/RUNBOOK-MULTI-NODE.md](docs/RUNBOOK-MULTI-NODE.md) | Two-node runbook proven on real hardware: every command from `node add` to `test-text`, measured memory/KV figures, and the failures worth knowing |
| [docs/FLEET-MULTI-NODE.md](docs/FLEET-MULTI-NODE.md) | Controlling several machines from one hub: registration, live resources, ConnectX/200G detection, cluster IPs |
| [docs/PRD.md](docs/PRD.md) | Product requirements, architecture, security, risks |
| [docs/CLI_SPEC.md](docs/CLI_SPEC.md) | CLI specification (unimplemented parts marked ❌) |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Milestones and phases |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup, invariants, how to add a preset/provider/gate |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## License

**Proprietary — all rights reserved.** See [LICENSE](LICENSE).

Bundles you generate belong to you and may be used, modified and redistributed freely. Model weights,
container images and third-party runtimes are covered by their own licenses.
