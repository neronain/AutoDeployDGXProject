#!/usr/bin/env python3
"""
gpu_util_rank_probe.py — is the gpu-util ceiling on DGX Spark (GB10) a function of
rank count, of the checkpoint, or only of host headroom?

Settles the open conflict in docs/DGX-SPARK-VLLM-FIELD-NOTES.md section 6, where one
third-party README says 0.75 at TP8 (DeepSeek-V4.1-Flash, "~24 GiB per rank for NCCL,
outside vLLM's budget") and another README from the same author says 0.80 at TP8
(GLM-5.3).  Neither ships a log.  Procedure, model choice, pre-registered decision
rules and artifact layout: docs/GPU-UTIL-RANK-PROTOCOL.md.

WHAT IT MEASURES
----------------
On unified memory there is exactly one pool and /proc/meminfo sees all of it, so the
quantity both READMEs are really arguing about is directly observable:

    X = (used while serving) - (used while idle) - (Model loading took + Available KV)

X is everything the engine put on the box OUTSIDE the budget vLLM thinks it manages.
If the "24 GiB per rank" story is right, X grows with world size.  If it is really the
MoE/NVFP4 load path (the COW break of docs/UPGRADE-2026-09.md 1.7 R8), X is large even
at N=1, where there is no NCCL at all.  If X is large but flat, the host-headroom rule
is right and 0.75/0.80 are both by-products of (MemTotal - X - safety) / MemTotal.

gpu-util is held FIXED in every cell.  This script never sweeps it to find the OOM
edge.  Sweeping-to-OOM is what the source repos did and it cannot separate "where it
broke" from "why"; the maximum is derived arithmetically from X instead.

PROVENANCE
----------
The claims under test ("~24 GiB per rank outside vLLM's budget", 0.75 at TP8, 0.80 at
TP8) come from two third-party READMEs, im0xMagnus/deepseek-v4.1-flash-uncensored-8x-
dgx-spark and im0xMagnus/glm-5.3-uncensored-8x-dgx-spark, MIT licensed, Copyright (c)
2026 im0xMagnus.  The 0.80-at-TP4 datapoint they are weighed against comes from
Tech2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark boot 10, which does commit head log,
args.json and kv-context.txt.  Those figures are cited as facts; no source text is
reproduced and no code from those repos is used here.  The ssh-the-source-over-stdin
idiom, the burn gate hand-off, the interleave discipline and the artifact tree follow
this repo's own scripts/hca_ab_bench.py and docs/HCA-DUAL-TEST.md.

This file is part of LMDS.  See the repository LICENSE.

WHERE IT RUNS
-------------
`drive` runs ON THE LMDS HUB that owns the target nodes -- the machine where plain
`lmds node ctl <node> ...` already works.  It is NOT runnable from a laptop that has
no fleet access.  Every other subcommand runs on a node and is normally invoked by
`drive` over ssh; it installs nothing, it is piped to `python3 -`.

USAGE
-----
    # 0. see the whole plan and every command, touching nothing
    python3 scripts/gpu_util_rank_probe.py plan \
        --arm D1:qwen3-8b:1 --arm D2:qwen3-8b:2 --arm M1:qwen3-5-122b-nvfp4:1 \
        --node-order msi-1,msi-2 --gpu-util 0.70 --repeats 3

    # 1. phase 0 -- one machine, no fabric, falsifies "24 GiB/rank is NCCL" on its own
    python3 scripts/gpu_util_rank_probe.py drive \
        --arm D1:qwen3-8b:1 --arm M1:qwen3-5-122b-nvfp4:1 \
        --node-order msi-1 \
        --ssh-target msi-1=neronain@10.10.1.1 \
        --gpu-util 0.70 --repeats 3 --out docs/evidence/gpu-util-rank/

    # 2. phase 1 -- the rank axis, same checkpoint at N=1 and N=2
    python3 scripts/gpu_util_rank_probe.py drive \
        --arm D1:qwen3-8b:1 --arm D2:qwen3-8b:2 \
        --node-order msi-1,msi-2 \
        --ssh-target msi-1=neronain@10.10.1.1 --ssh-target msi-2=neronain@10.10.1.2 \
        --gpu-util 0.70 --repeats 3 --out docs/evidence/gpu-util-rank/

    # re-derive the verdict later without touching the fleet
    python3 scripts/gpu_util_rank_probe.py verdict --results <dir>/results.json

    # pieces, on a node
    python3 scripts/gpu_util_rank_probe.py snapshot
    python3 scripts/gpu_util_rank_probe.py idle
    python3 scripts/gpu_util_rank_probe.py watch --seconds 600
"""

import argparse
import json
import os
import pathlib
import re
import shlex
import statistics
import subprocess
import sys
import time

GIB = 1024 ** 3

# ---- pre-registered thresholds.  Written before the first run; do not tune to fit
#      a result.  docs/GPU-UTIL-RANK-PROTOCOL.md section 5 is the normative copy.
MIN_REPEATS = 3
IDLE_DRIFT_ABORT_GIB = 2.0     # A-B-A: idle baseline must return to where it started
RANK_EFFECT_MIN_GIB = 2.0      # dX_rank at or above this = the rank term is real
CKPT_EFFECT_MIN_GIB = 5.0      # dX_ckpt at or above this = the checkpoint term is real
CKPT_DOMINANCE = 2.0           # dX_ckpt >= this * dX_rank = checkpoint wins
FLAT_X_MIN_GIB = 8.0           # X this big but flat = host-headroom rule
NO_NCCL_AT_N1_GIB = 15.0       # X(N=1) at or above this falsifies "24 GiB/rank is NCCL"
SAFETY_MARGIN_GIB = 6.0        # headroom that must survive the load peak
AGREEMENT_TOLERANCE_GIB = 3.0  # x_accounted vs x_budget must agree, or the cell is suspect

# Nodes this script must never touch.  dgx-70 carries a customer's model: it is never
# started, never stopped, never measured.  See docs/FLEET-STATE.md section 7.
DEFAULT_DENY = ("dgx-70",)

# vLLM start lines.  Same regexes as src/lmds/fit/sizing.py so the two agree by
# construction.  Token counts are deliberately NOT a decision variable:
# docs/DGX-SPARK-VLLM-FIELD-NOTES.md section 11 -- vLLM prints
# max_concurrency * max_model_len, which moves with --max-model-len at constant memory.
_LOADING_RE = re.compile(r"Model loading took ([\d.]+) GiB")
_AVAILABLE_RE = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
_RESERVED_RE = re.compile(r"reserved ([\d.]+) GiB memory for KV Cache")
_KV_TOKENS_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_NCCL_IF_RE = re.compile(r"NCCL if\s*:\s*(\S+)")
_NCCL_HCA_RE = re.compile(r"NCCL HCA\s*:\s*(\S+)")


# ======================================================================= node side

def _read(path, default=""):
    try:
        return pathlib.Path(path).read_text()
    except OSError:
        return default


def meminfo():
    """/proc/meminfo in GiB.  On GB10 this pool is the GPU pool -- one number, not two."""
    out = {}
    for line in _read("/proc/meminfo").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            try:
                out[parts[0][:-1]] = int(parts[1]) / (1024 * 1024)  # kB -> GiB
            except ValueError:
                pass
    return out


def engine_proc():
    """The vLLM process and its COW footprint, or None.

    Private_Dirty is the direct read on the docs/UPGRADE-2026-09.md 1.7 R8 mechanism:
    a fused-MoE expert copy breaks the MAP_PRIVATE mapping of every shard it touches
    and the pages become anonymous ones the kernel cannot reclaim.  If the disputed
    "~24 GiB per rank" is that, it shows up here and NOT in anything NCCL does.
    """
    best = None
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        cmd = _read(f"/proc/{entry.name}/cmdline").replace("\0", " ")
        if "vllm" not in cmd.lower():
            continue
        roll = _read(f"/proc/{entry.name}/smaps_rollup")
        vals = {}
        for line in roll.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].endswith(":"):
                try:
                    vals[parts[0][:-1]] = int(parts[1]) / (1024 * 1024)
                except ValueError:
                    pass
        rss = vals.get("Rss", 0.0)
        if best is None or rss > best["rss_gib"]:
            best = {
                "pid": int(entry.name),
                "cmdline": cmd.strip()[:400],
                "rss_gib": round(rss, 2),
                "private_dirty_gib": round(vals.get("Private_Dirty", 0.0), 2),
                "private_clean_gib": round(vals.get("Private_Clean", 0.0), 2),
                "shared_clean_gib": round(vals.get("Shared_Clean", 0.0), 2),
            }
    return best


def _cmd(argv, timeout=20):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def cmd_snapshot(args):
    mi = meminfo()
    total = mi.get("MemTotal", 0.0)
    avail = mi.get("MemAvailable", 0.0)
    snap = {
        "host": os.uname().nodename,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": args.label,
        "mem_total_gib": round(total, 2),
        "mem_available_gib": round(avail, 2),
        "used_gib": round(total - avail, 2),
        "anon_pages_gib": round(mi.get("AnonPages", 0.0), 2),
        "cached_gib": round(mi.get("Cached", 0.0), 2),
        "shmem_gib": round(mi.get("Shmem", 0.0), 2),
        "mapped_gib": round(mi.get("Mapped", 0.0), 2),
        "engine": engine_proc(),
        "nvidia_smi_mem": _cmd(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                "--format=csv,noheader,nounits"]),
        "nvidia_smi_apps": _cmd(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                 "--format=csv,noheader"]),
        "docker_ps": _cmd(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"]),
    }
    print("SNAPSHOT " + json.dumps(snap))
    return 0


def cmd_idle(args):
    """Idle baseline, and a refusal to measure a node that is not idle.

    A node with somebody else's model resident has a different X and a different page
    cache, and the cell would be uninterpretable.  This never stops anything: it
    reports and exits non-zero so the driver aborts and a human decides.
    """
    mi = meminfo()
    total, avail = mi.get("MemTotal", 0.0), mi.get("MemAvailable", 0.0)
    apps = _cmd(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"])
    docker = _cmd(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"])
    busy = [x for x in (apps.splitlines() if apps else []) if x.strip()]
    containers = [x for x in (docker.splitlines() if docker else []) if x.strip()]
    engine = engine_proc()
    ok = not busy and not containers and engine is None
    print("IDLE " + json.dumps({
        "host": os.uname().nodename,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mem_total_gib": round(total, 2),
        "mem_available_gib": round(avail, 2),
        "idle_used_gib": round(total - avail, 2),
        "gpu_apps": busy,
        "containers": containers,
        "engine": engine,
        "idle": ok,
    }))
    return 0 if ok else 3


def cmd_watch(args):
    """Poll meminfo and keep the worst moment.

    The load peak, not the steady state, is where a stacked boot dies.  Sampling it is
    the only way to say how much of the headroom the measurement actually consumed.
    """
    total = meminfo().get("MemTotal", 0.0)
    worst = None
    t0 = time.time()
    samples = 0
    while time.time() - t0 < args.seconds:
        avail = meminfo().get("MemAvailable", 0.0)
        samples += 1
        if worst is None or avail < worst:
            worst = avail
        if args.stop_file and pathlib.Path(args.stop_file).exists():
            break
        time.sleep(args.interval)
    print("WATCH " + json.dumps({
        "host": os.uname().nodename,
        "mem_total_gib": round(total, 2),
        "min_available_gib": round(worst if worst is not None else 0.0, 2),
        "peak_used_gib": round(total - (worst if worst is not None else 0.0), 2),
        "samples": samples,
        "seconds": round(time.time() - t0, 1),
    }))
    return 0


# ======================================================================== hub side

def _run(argv, timeout=1800, check=False):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise SystemExit(f"ABORT: command failed ({p.returncode}): {' '.join(argv)}\n"
                         f"{p.stdout}\n{p.stderr}")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _ssh_py(target, argv, source, timeout=1800, background=False):
    """Run this script on `target` by piping its source to python3 -- installs nothing."""
    remote = "python3 - " + " ".join(shlex.quote(a) for a in argv)
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target, remote]
    if background:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        p.stdin.write(source)
        p.stdin.close()
        return p
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    out, _ = p.communicate(source, timeout=timeout)
    return p.returncode, out


def _tagged(text, tag):
    for line in (text or "").splitlines():
        if line.startswith(tag + " "):
            return json.loads(line[len(tag) + 1:])
    return None


def _lmds(args, argv, timeout=1800):
    return _run([args.lmds] + argv, timeout=timeout)


def parse_arms(specs):
    arms = []
    for spec in specs:
        try:
            name, slug, nodes = spec.split(":")
            arms.append({"name": name, "slug": slug, "nodes": int(nodes)})
        except ValueError as exc:
            raise SystemExit(f"ABORT: --arm wants NAME:SLUG:NODES, got {spec!r}") from exc
    names = [a["name"] for a in arms]
    if len(set(names)) != len(names):
        raise SystemExit("ABORT: duplicate arm names")
    return arms


def parse_targets(specs):
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"ABORT: --ssh-target wants NAME=user@host, got {spec!r}")
        name, target = spec.split("=", 1)
        out[name] = target
    return out


def measured_from_log(text):
    out = {}
    for key, rx in (("loading_took_gib", _LOADING_RE),
                    ("available_kv_gib", _AVAILABLE_RE),
                    ("reserved_kv_gib", _RESERVED_RE)):
        found = rx.findall(text or "")
        if found:
            out[key] = float(found[-1])
    tokens = _KV_TOKENS_RE.findall(text or "")
    if tokens:
        # recorded for the record only -- never a decision variable (section 11)
        out["kv_cache_tokens_DO_NOT_USE"] = int(tokens[-1].replace(",", ""))
    pool = out.get("reserved_kv_gib") or out.get("available_kv_gib")
    if pool:
        out["kv_pool_gib"] = pool
    return out


# ------------------------------------------------------------------------ preflight

def check_denylist(nodes, deny):
    hit = sorted(set(nodes) & set(deny))
    if hit:
        raise SystemExit(
            f"ABORT: {', '.join(hit)} is on the deny list and carries a customer model. "
            "It is never started, stopped or measured.  Pick other nodes.")


def burn_gate(args, node, outdir, when):
    """Hand off to the burn check -- this script does not implement one.

    docs/DGX-SPARK-VLLM-FIELD-NOTES.md section 10: the EC can latch a GB10 at 631-949
    MHz with nvidia-smi reporting nothing wrong, and in a collective every rank waits
    for the slowest.  A latched node does not merely add noise, it silently changes
    the thing being measured.  Owned by the burn-gate work stream; when `lmds burn`
    ships, point --burn-cmd at it.  A missing burn gate is an ABORT, not a warning.
    """
    cmd = [x.replace("{node}", node) for x in shlex.split(args.burn_cmd)]
    rc, out = _run(cmd, timeout=300)
    (outdir / "burn.txt").open("a").write(
        f"--- {when} {node} rc={rc}\n{out}\n")
    print(f"[burn:{when}:{node}] rc={rc}")
    if rc != 0 and not args.skip_burn:
        raise SystemExit(
            f"ABORT: {node} failed the GPU burn gate ({when}).  A latched node caps every\n"
            "rank and every number in this session would be a lie.  Fix it -- power off\n"
            "and UNPLUG the adapter for 30-60 s; a normal reboot does not clear it -- then\n"
            "re-run.  --skip-burn overrides and permanently taints the result.")
    return {"rc": rc, "out": out[-4000:]}


def idle_baseline(args, node, source):
    target = args.ssh[node]
    rc, out = _ssh_py(target, ["idle"], source, timeout=120)
    data = _tagged(out, "IDLE")
    if data is None:
        raise SystemExit(f"ABORT: could not read the idle baseline on {node}.\n{out}")
    if not data["idle"] and not args.allow_busy:
        raise SystemExit(
            f"ABORT: {node} is not idle.\n"
            f"  gpu apps : {data['gpu_apps']}\n"
            f"  containers: {data['containers']}\n"
            "This script never stops what it did not start.  Free the node by hand, or\n"
            "pick another one.  --allow-busy overrides and taints every cell on it.")
    return data


# ----------------------------------------------------------------------- cell

def run_cell(args, arm, rep, nodes, source, outdir):
    """One (arm, repeat).  Exactly one thing differs between cells: arm -- which is
    (checkpoint, node count).  gpu-util, image, context and prompt are identical
    everywhere.  The source repos changed gpu-util and tools/vision in the same boot
    and then attributed the difference to a third thing; this does not.
    """
    slug, n = arm["slug"], arm["nodes"]
    used = nodes[:n]
    head = used[0]
    tag = f"{arm['name']}-rep{rep}"
    raw = outdir / "raw"
    cell = {"arm": arm["name"], "slug": slug, "nodes": n, "node_names": used,
            "repeat": rep, "gpu_util": args.gpu_util,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "per_node": {}, "status": "running", "notes": []}

    # -- pre snapshot, every node in the cell
    pre = {}
    for node in used:
        rc, out = _ssh_py(args.ssh[node], ["snapshot", "--label", f"{tag}-pre"], source, timeout=120)
        snap = _tagged(out, "SNAPSHOT")
        if snap is None:
            cell["status"] = "abort:pre-snapshot"
            return cell
        pre[node] = snap
        (raw / f"{tag}-{node}-pre.json").write_text(json.dumps(snap, indent=2))

    # -- cluster wiring for N>=2, then prove the fabric is really the fabric
    if n >= 2:
        rc, out = _lmds(args, ["cluster", "write", slug, "--head", head], timeout=300)
        (raw / f"{tag}-cluster-write.txt").write_text(out)
        if rc != 0:
            cell["status"] = "abort:cluster-write"
            cell["notes"].append(out[-2000:])
            return cell
        # Advisory only at this point -- on some controller versions the fabric does
        # not resolve until the container exists.  The authoritative check is the one
        # after start, below.
        _fabric_check(args, head, slug, raw, tag, cell, "pre", fatal=False)

    # -- watchers, then start
    stop_file = f"/tmp/lmds-gur-{tag}.stop"
    watchers = {}
    for node in used:
        _run(["ssh", "-o", "BatchMode=yes", args.ssh[node], f"rm -f {shlex.quote(stop_file)}"],
             timeout=60)
        watchers[node] = _ssh_py(args.ssh[node],
                                 ["watch", "--seconds", str(args.watch_seconds),
                                  "--interval", "2", "--stop-file", stop_file],
                                 source, background=True)

    t_start = time.time()
    rc, out = _lmds(args, ["node", "ctl", head, slug, "start",
                           "--gpu-util", str(args.gpu_util)], timeout=args.start_timeout)
    cell["start_rc"] = rc
    cell["start_seconds"] = round(time.time() - t_start, 1)
    (raw / f"{tag}-start.txt").write_text(out)

    if rc == 0 and n >= 2:
        # Authoritative: it is serving now, so the fabric it is serving on is knowable.
        if not _fabric_check(args, head, slug, raw, tag, cell, "post", fatal=True):
            _lmds(args, ["node", "run", head, "stop", slug], timeout=900)
            for node in used:
                _run(["ssh", "-o", "BatchMode=yes", args.ssh[node],
                      f"touch {shlex.quote(stop_file)}"], timeout=60)
            for p in watchers.values():
                try:
                    p.communicate(timeout=120)
                except subprocess.TimeoutExpired:
                    p.kill()
            return cell

    if rc == 0:
        rc_t, out_t = _lmds(args, ["node", "ctl", head, slug, "test-text"], timeout=900)
        cell["test_text_rc"] = rc_t
        (raw / f"{tag}-test-text.txt").write_text(out_t)

        # -- steady snapshot while it is actually serving
        time.sleep(args.settle_seconds)
        for node in used:
            rc_s, out_s = _ssh_py(args.ssh[node], ["snapshot", "--label", f"{tag}-steady"],
                                  source, timeout=120)
            snap = _tagged(out_s, "SNAPSHOT")
            if snap:
                (raw / f"{tag}-{node}-steady.json").write_text(json.dumps(snap, indent=2))
                cell["per_node"].setdefault(node, {})["steady"] = snap
    else:
        cell["notes"].append("start failed -- kept as evidence, not averaged in")

    # -- stop the watchers and collect the load peak
    for node in used:
        _run(["ssh", "-o", "BatchMode=yes", args.ssh[node], f"touch {shlex.quote(stop_file)}"],
             timeout=60)
    for node, p in watchers.items():
        try:
            out_w, _ = p.communicate(timeout=args.watch_seconds + 120)
        except subprocess.TimeoutExpired:
            p.kill()
            out_w = ""
        w = _tagged(out_w, "WATCH")
        if w:
            cell["per_node"].setdefault(node, {})["peak"] = w
        (raw / f"{tag}-{node}-watch.txt").write_text(out_w or "")

    # -- engine logs, every rank.  Keep them even for a failed boot: section 1.9 of
    #    UPGRADE-2026-09 notes we throw away the evidence of every boot that dies.
    for node in used:
        rc_l, out_l = _lmds(args, ["node", "run", node, "logs", slug, "-n", "600"], timeout=300)
        (raw / f"{tag}-{node}-engine.log").write_text(out_l)
        cell["per_node"].setdefault(node, {})["log"] = measured_from_log(out_l)

    # -- stop only what we started
    rc_x, out_x = _lmds(args, ["node", "run", head, "stop", slug], timeout=900)
    (raw / f"{tag}-stop.txt").write_text(out_x)
    time.sleep(args.settle_seconds)

    # -- back to idle?  If not, every later cell inherits the drift.
    for node in used:
        rc_i, out_i = _ssh_py(args.ssh[node], ["idle"], source, timeout=120)
        post = _tagged(out_i, "IDLE")
        cell["per_node"].setdefault(node, {})["post_idle"] = post
        cell["per_node"][node]["pre_idle_used_gib"] = pre[node]["used_gib"]
        if post and not post["idle"]:
            cell["notes"].append(f"{node} did not return to idle after stop")

    cell["status"] = "ok" if cell.get("start_rc") == 0 else "failed-boot"
    _derive_cell(cell)
    return cell


def _fabric_check(args, head, slug, raw, tag, cell, when, fatal):
    """Is NCCL really on the RDMA fabric?

    RUNBOOK-MULTI-NODE section 5: an empty interface or HCA means NCCL fell back to the
    management link or plain TCP.  It still serves, /health still passes, nothing
    complains -- and the memory footprint is not the footprint of an RDMA run, so the
    cell would be measuring a different thing under the same label.
    """
    rc, out = _lmds(args, ["node", "ctl", head, slug, "network-info"], timeout=300)
    (raw / f"{tag}-network-info-{when}.txt").write_text(out)
    m_if, m_hca = _NCCL_IF_RE.search(out), _NCCL_HCA_RE.search(out)
    cell[f"nccl_if_{when}"] = m_if.group(1) if m_if else None
    cell[f"nccl_hca_{when}"] = m_hca.group(1) if m_hca else None
    ok = bool(cell[f"nccl_if_{when}"] and cell[f"nccl_hca_{when}"])
    if not ok:
        note = f"network-info ({when}) reported no NCCL interface or HCA"
        cell["notes"].append(note if fatal else note + " -- advisory, see the post check")
        if fatal:
            cell["status"] = "abort:no-fabric"
    return ok


def _derive_cell(cell):
    """X per node, two independent ways, and the disagreement between them.

    x_accounted is primary: serving used, minus idle used, minus the two numbers vLLM
    itself prints.  Whatever is left is on the box but outside vLLM's accounting.
    x_budget is the cross-check against the budget vLLM was given.  They should agree;
    when they do not, the cell is flagged rather than quietly averaged.
    """
    for node, d in cell["per_node"].items():
        steady = d.get("steady")
        log = d.get("log") or {}
        idle_used = d.get("pre_idle_used_gib")
        if not steady or idle_used is None:
            continue
        total = steady["mem_total_gib"]
        used = steady["used_gib"]
        acc = log.get("loading_took_gib"), log.get("available_kv_gib")
        if all(v is not None for v in acc):
            d["x_accounted_gib"] = round(used - idle_used - acc[0] - acc[1], 2)
        else:
            d["x_accounted_gib"] = None
            cell["notes"].append(
                f"{node}: no 'Model loading took' / 'Available KV cache memory' line -- "
                "the cell has no primary measurement")
        d["x_budget_gib"] = round(used - idle_used - cell["gpu_util"] * total, 2)
        if d["x_accounted_gib"] is not None:
            d["x_disagreement_gib"] = round(abs(d["x_accounted_gib"] - d["x_budget_gib"]), 2)
            if d["x_disagreement_gib"] > AGREEMENT_TOLERANCE_GIB:
                cell["notes"].append(
                    f"{node}: x_accounted and x_budget differ by "
                    f"{d['x_disagreement_gib']} GiB -- vLLM did not fill its budget, or the "
                    "log and the box disagree.  Cell is suspect.")
        peak = d.get("peak") or {}
        if peak.get("min_available_gib") is not None:
            d["peak_headroom_gib"] = peak["min_available_gib"]
    xs = [d["x_accounted_gib"] for d in cell["per_node"].values()
          if d.get("x_accounted_gib") is not None]
    cell["x_gib"] = round(max(xs), 2) if xs else None   # the binding node
    heads = [d.get("peak_headroom_gib") for d in cell["per_node"].values()
             if d.get("peak_headroom_gib") is not None]
    cell["peak_headroom_gib"] = round(min(heads), 2) if heads else None
    pds = [(d.get("steady") or {}).get("engine", {}).get("private_dirty_gib")
           for d in cell["per_node"].values()
           if (d.get("steady") or {}).get("engine")]
    pds = [p for p in pds if p is not None]
    cell["private_dirty_gib"] = round(max(pds), 2) if pds else None


# --------------------------------------------------------------------- verdict

def _stats(values):
    if not values:
        return None
    return {"n": len(values), "median": round(statistics.median(values), 2),
            "min": round(min(values), 2), "max": round(max(values), 2),
            "spread": round((max(values) - min(values)) / statistics.median(values), 3)
            if statistics.median(values) else None}


def derive_verdict(cells, meta=None):
    """The pre-registered rules.  Written before the first run.

    This function is the normative implementation of section 5 of the protocol doc.
    It is deliberately callable on an existing results.json so the decision can be
    re-derived by anyone without re-running anything, and so it can be diffed against
    the version that existed before the data did.
    """
    meta = meta or {}
    counted = [c for c in cells if c.get("status") == "ok" and c.get("x_gib") is not None]
    by_arm = {}
    for c in counted:
        by_arm.setdefault(c["arm"], []).append(c["x_gib"])
    stats = {arm: _stats(v) for arm, v in by_arm.items()}

    blockers = []
    if meta.get("burn_failed"):
        blockers.append("a node failed the GPU burn gate -- clock latch invalidates every cell")
    if meta.get("idle_drift_gib") is not None and meta["idle_drift_gib"] > IDLE_DRIFT_ABORT_GIB:
        blockers.append(
            f"idle baseline drifted {meta['idle_drift_gib']} GiB between the first and last "
            f"reading (limit {IDLE_DRIFT_ABORT_GIB}) -- the session, not the arms, moved")
    for arm, s in stats.items():
        if s["n"] < MIN_REPEATS:
            blockers.append(f"arm {arm} has {s['n']} valid repeats, needs {MIN_REPEATS}")
    missing = [c["arm"] for c in cells if c.get("status") == "ok" and c.get("x_gib") is None]
    if missing:
        blockers.append(f"arms with no primary measurement (missing vLLM GiB lines): "
                        f"{sorted(set(missing))}")

    # the arms, by role.  N=1 arms named D*/M* by convention; roles are resolved from
    # the cells themselves so the naming is a convenience, not a contract.
    def arm_nodes(arm):
        for c in cells:
            if c["arm"] == arm:
                return c["nodes"], c["slug"]
        return None, None

    dense_n1 = dense_n2 = moe_n1 = None
    slugs_n1 = {}
    for arm in stats:
        n, slug = arm_nodes(arm)
        if n == 1:
            slugs_n1.setdefault(slug, arm)
    # the rank axis is the one slug that appears at two different node counts
    for arm in stats:
        n, slug = arm_nodes(arm)
        if n == 1 and any(arm_nodes(a) == (2, slug) for a in stats):
            dense_n1, dense_n2 = arm, next(a for a in stats if arm_nodes(a) == (2, slug))
    if dense_n1:
        base_slug = arm_nodes(dense_n1)[1]
        for slug, arm in slugs_n1.items():
            if slug != base_slug:
                moe_n1 = arm

    out = {"stats": stats, "blockers": blockers,
           "rank_axis": [dense_n1, dense_n2], "checkpoint_axis": [dense_n1, moe_n1],
           "thresholds": {
               "RANK_EFFECT_MIN_GIB": RANK_EFFECT_MIN_GIB,
               "CKPT_EFFECT_MIN_GIB": CKPT_EFFECT_MIN_GIB,
               "CKPT_DOMINANCE": CKPT_DOMINANCE,
               "FLAT_X_MIN_GIB": FLAT_X_MIN_GIB,
               "NO_NCCL_AT_N1_GIB": NO_NCCL_AT_N1_GIB,
           }}

    dx_rank = dx_ckpt = None
    if dense_n1 and dense_n2:
        dx_rank = round(stats[dense_n2]["median"] - stats[dense_n1]["median"], 2)
        out["dx_rank_gib"] = dx_rank
        out["rank_separated"] = stats[dense_n2]["min"] > stats[dense_n1]["max"]
    if dense_n1 and moe_n1:
        dx_ckpt = round(stats[moe_n1]["median"] - stats[dense_n1]["median"], 2)
        out["dx_ckpt_gib"] = dx_ckpt
        out["ckpt_separated"] = stats[moe_n1]["min"] > stats[dense_n1]["max"]

    # standalone falsification -- no comparison needed.  At N=1 there is no NCCL at all.
    n1_max = max([s["median"] for a, s in stats.items() if arm_nodes(a)[0] == 1] or [0])
    out["x_at_n1_gib"] = round(n1_max, 2)
    if n1_max >= NO_NCCL_AT_N1_GIB:
        out["falsified"] = (
            f"X at N=1 is {n1_max:.1f} GiB with no NCCL in the process at all.  A large "
            "out-of-budget footprint therefore is NOT evidence for a per-rank NCCL "
            "reserve, and the '~24 GiB per rank' attribution does not survive.")

    if blockers:
        out["verdict"] = "INVALID"
        out["reason"] = "; ".join(blockers)
        return out

    if dx_rank is not None and dx_ckpt is not None:
        if dx_ckpt >= CKPT_EFFECT_MIN_GIB and dx_ckpt >= CKPT_DOMINANCE * max(dx_rank, 0.01) \
                and out.get("ckpt_separated"):
            out["verdict"] = "C"
            out["reason"] = (
                f"the checkpoint moves X by {dx_ckpt} GiB, the rank count by {dx_rank} GiB.  "
                "The two contested READMEs ran different models, so both numbers can be right "
                "for their own checkpoint and neither is a function of 8.  fit must NOT gain a "
                "rank-scaled NCCL term; it needs a per-checkpoint host-footprint term.")
            return out
        if dx_rank >= RANK_EFFECT_MIN_GIB and out.get("rank_separated") and dx_rank > dx_ckpt:
            out["verdict"] = "R"
            out["reason"] = (
                f"adding one rank moves X by {dx_rank} GiB with non-overlapping ranges, more "
                f"than the checkpoint's {dx_ckpt} GiB.  The rank term is real over N=1..2.")
            out["extrapolation"] = _extrapolate(stats[dense_n1]["median"], dx_rank)
            return out
    elif dx_rank is not None and dx_rank >= RANK_EFFECT_MIN_GIB and out.get("rank_separated"):
        out["verdict"] = "R"
        out["reason"] = (f"adding one rank moves X by {dx_rank} GiB with non-overlapping "
                         "ranges; no checkpoint arm was run, so the checkpoint term is untested.")
        out["extrapolation"] = _extrapolate(stats[dense_n1]["median"], dx_rank)
        return out

    all_med = [s["median"] for s in stats.values()]
    flat = all_med and (max(all_med) - min(all_med)) < RANK_EFFECT_MIN_GIB
    if flat and min(all_med) >= FLAT_X_MIN_GIB:
        out["verdict"] = "H"
        out["reason"] = (
            f"X is {min(all_med):.1f}-{max(all_med):.1f} GiB in every arm and moves with neither "
            "rank count nor checkpoint.  The ceiling is host headroom: 0.75 and 0.80 are both "
            "by-products of (MemTotal - X - safety) / MemTotal on boxes with slightly different "
            "X.  fit gets a measured host reserve and no rank term.")
        return out

    out["verdict"] = "INCONCLUSIVE"
    out["reason"] = (
        "no effect cleared its threshold with separated ranges.  Most likely the arms are too "
        "close together to resolve at this repeat count.  Next: raise --repeats, or widen the "
        "checkpoint axis (a much larger MoE at N=1), before adding node count.")
    return out


def _extrapolate(x1, dx):
    """State an N=8 number ONLY as a labelled extrapolation.

    Two points cannot establish linearity, and NCCL changes shape above N=2 (rings and
    trees, more channels, cross-cage routing).  This is a prediction to be falsified by
    a real 8-node run, not a measurement.
    """
    rows = {}
    for n in (2, 4, 8):
        x = x1 + dx * (n - 1)
        rows[f"N={n}"] = {
            "x_predicted_gib": round(x, 2),
            "gpu_util_max": round(max(0.0, (121.7 - x - SAFETY_MARGIN_GIB) / 121.7), 3),
        }
    rows["_warning"] = ("EXTRAPOLATION from two points (N=1, N=2).  Not a measurement. "
                        "Do not write it into a catalog as validated.")
    return rows


def _fit_input(cells, verdict, meta):
    """The hand-off to src/lmds/fit/ -- field names match the constants they replace."""
    idle = [v for v in (meta.get("idle_used_gib") or {}).values() if v is not None]
    stats = verdict.get("stats") or {}
    out = {
        "source": "scripts/gpu_util_rank_probe.py",
        "protocol": "docs/GPU-UTIL-RANK-PROTOCOL.md",
        "verdict": verdict.get("verdict"),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # analyzer.py:50 -- UNIFIED_OS_RESERVE_GB = 12.0, a guess until now.
        "UNIFIED_OS_RESERVE_GB_measured": round(max(idle), 2) if idle else None,
        "UNIFIED_OS_RESERVE_GB_current": 12.0,
        # analyzer.py:60 -- STACKED_COMM_BUFFER_GB_PER_NODE = 3.0, "~2-3 GB per rank".
        "STACKED_COMM_BUFFER_GB_PER_NODE_measured": verdict.get("dx_rank_gib"),
        "STACKED_COMM_BUFFER_GB_PER_NODE_current": 3.0,
        "STACKED_COMM_BUFFER_valid_for_node_counts": [1, 2],
        # sizing.py -- read GiB, never tokens (field notes section 11).
        "kv_read_from": "Available KV cache memory (GiB)",
        "per_arm_x_gib": stats,
        "checkpoint_host_footprint_gib": verdict.get("dx_ckpt_gib"),
        "x_at_n1_gib": verdict.get("x_at_n1_gib"),
        "extrapolation_do_not_treat_as_measured": verdict.get("extrapolation"),
        "cells": [{"arm": c["arm"], "slug": c["slug"], "nodes": c["nodes"],
                   "repeat": c["repeat"], "gpu_util": c["gpu_util"],
                   "x_gib": c.get("x_gib"), "peak_headroom_gib": c.get("peak_headroom_gib"),
                   "private_dirty_gib": c.get("private_dirty_gib"),
                   "status": c["status"]} for c in cells],
    }
    return out


def _render_readme(args, cells, verdict, meta):
    counted = [c for c in cells if c["status"] == "ok"]
    lines = [
        f"# gpu-util rank probe — {', '.join(args.node_order)}",
        "",
        f"**Verdict: {verdict.get('verdict')}** — {verdict.get('reason','')}",
        "",
        "- protocol: `docs/GPU-UTIL-RANK-PROTOCOL.md` (decision rules were written before this run)",
        f"- gpu-util held fixed at **{args.gpu_util}** in every cell",
        f"- cells: {len(cells)} run, {len(counted)} counted, {args.repeats} repeats, interleaved",
        f"- idle drift between first and last baseline: {meta.get('idle_drift_gib')} GiB "
        f"(abort above {IDLE_DRIFT_ABORT_GIB})",
        "",
        "| arm | slug | N | X median GiB | min | max | spread |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, s in (verdict.get("stats") or {}).items():
        c = next((x for x in cells if x["arm"] == arm), {})
        lines.append(f"| {arm} | `{c.get('slug','')}` | {c.get('nodes','')} | "
                     f"{s['median']} | {s['min']} | {s['max']} | {s['spread']} |")
    lines += [
        "",
        f"- `dx_rank` (same checkpoint, +1 node): **{verdict.get('dx_rank_gib')} GiB**",
        f"- `dx_ckpt` (same node count, other checkpoint): **{verdict.get('dx_ckpt_gib')} GiB**",
        f"- X at N=1, where there is no NCCL at all: **{verdict.get('x_at_n1_gib')} GiB**",
        "",
    ]
    if verdict.get("falsified"):
        lines += ["> " + verdict["falsified"], ""]
    lines += [
        "## What this does not answer",
        "",
        "- Nothing about N=4 or N=8.  Two points cannot establish linearity and NCCL changes",
        "  shape above two ranks.  Any N=8 figure in `fit-input.json` is labelled as an",
        "  extrapolation and must not be written into a catalog as validated.",
        "- Nothing about DeepSeek-V4.1-Flash or GLM-5.3 specifically — neither was run.",
        "- Nothing about the 500K–900K prefill regime the 0.75 claim came from.",
        "",
        "## Files",
        "",
        "`results.json` every cell · `fit-input.json` the hand-off to `src/lmds/fit/` ·",
        "`burn.txt` clock-latch gate before and after · `idle.txt` baselines ·",
        "`args.json` real argv · `raw/` unfiltered engine logs and snapshots for every cell,",
        "**including the ones that were discarded** — a boot that died is evidence about the fleet.",
    ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- drive

def cmd_plan(args):
    arms = parse_arms(args.arm)
    order = args.node_order
    per_cell_min = {1: 8, 2: 12}
    total = 0
    print(f"hub       : run this where plain `{args.lmds} node ctl <node> ...` already works")
    print(f"nodes     : {', '.join(order)}")
    print(f"deny list : {', '.join(args.deny)}  (never started, stopped or measured)")
    print(f"gpu-util  : {args.gpu_util} — FIXED in every cell, never swept")
    print(f"repeats   : {args.repeats} (interleaved, rotated each repeat)\n")
    print("order of cells:")
    for rep in range(1, args.repeats + 1):
        seq = arms[rep % len(arms):] + arms[:rep % len(arms)]
        names = []
        for a in seq:
            names.append(f"{a['name']}(N={a['nodes']})")
            total += per_cell_min.get(a["nodes"], 12)
        print(f"  repeat {rep}: " + "  ".join(names))
    print(f"\nestimated fleet time: ~{total} min of cells "
          f"+ ~20 min preflight/teardown = ~{(total + 20) / 60:.1f} h")
    print("  assumes every weight is already on every node "
          f"(`{args.lmds} node run <node> scan` first — a download adds hours)\n")
    print("commands per cell (N>=2 adds the first two):")
    print(f"  {args.lmds} cluster write <slug> --head <head>")
    print(f"  {args.lmds} node ctl <head> <slug> network-info        # abort if NCCL if/HCA empty")
    print(f"  {args.lmds} node ctl <head> <slug> start --gpu-util {args.gpu_util}")
    print(f"  {args.lmds} node ctl <head> <slug> test-text")
    print(f"  {args.lmds} node run <node> logs <slug> -n 600         # every rank")
    print(f"  {args.lmds} node run <head> stop <slug>                # only what we started")
    return 0


def cmd_drive(args):
    source = pathlib.Path(__file__).read_text()
    arms = parse_arms(args.arm)
    args.ssh = parse_targets(args.ssh_target)
    order = args.node_order
    check_denylist(order, args.deny)

    need = max(a["nodes"] for a in arms)
    if len(order) < need:
        raise SystemExit(f"ABORT: an arm wants {need} nodes, --node-order has {len(order)}.")
    missing = [n for n in order if n not in args.ssh]
    if missing:
        raise SystemExit(f"ABORT: no --ssh-target for {', '.join(missing)} "
                         f"(get user@host from `{args.lmds} node ls`)")
    if args.repeats < MIN_REPEATS:
        raise SystemExit(f"ABORT: --repeats must be at least {MIN_REPEATS}.  "
                         "One reading per cell against ~34% run-to-run spread is not a result.")

    stamp = time.strftime("%Y-%m-%d", time.gmtime())
    outdir = pathlib.Path(args.out) / f"{stamp}-{'-'.join(order[:need])}"
    (outdir / "raw").mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps(
        {"argv": sys.argv, "utc": stamp, "gpu_util": args.gpu_util,
         "arms": arms, "node_order": order, "repeats": args.repeats}, indent=2))

    meta = {"idle_used_gib": {}, "burn_failed": False}

    # ---- P1 burn gate, every node, before anything
    for node in order[:need]:
        burn_gate(args, node, outdir, "before")

    # ---- P2 idle baseline, every node
    first_idle = {}
    for node in order[:need]:
        d = idle_baseline(args, node, source)
        first_idle[node] = d
        meta["idle_used_gib"][node] = d["idle_used_gib"]
    (outdir / "idle.txt").write_text(json.dumps({"before": first_idle}, indent=2))

    # ---- interleaved sweep.  Rotate the arm order every repeat so a node that warms,
    #      drifts or latches mid-session cannot masquerade as an arm effect.
    cells = []
    for rep in range(1, args.repeats + 1):
        seq = arms[rep % len(arms):] + arms[:rep % len(arms)]
        for arm in seq:
            print(f"--- repeat {rep} arm {arm['name']} (slug {arm['slug']}, N={arm['nodes']}) ---")
            cell = run_cell(args, arm, rep, order, source, outdir)
            cells.append(cell)
            (outdir / "results.json").write_text(json.dumps(cells, indent=2))
            print(f"    status={cell['status']} X={cell.get('x_gib')} GiB "
                  f"peak_headroom={cell.get('peak_headroom_gib')} GiB")

    # ---- A-B-A: the idle baseline must come back to where it started
    last_idle = {}
    drift = 0.0
    for node in order[:need]:
        rc, out = _ssh_py(args.ssh[node], ["idle"], source, timeout=120)
        d = _tagged(out, "IDLE")
        last_idle[node] = d
        if d:
            drift = max(drift, abs(d["idle_used_gib"] - first_idle[node]["idle_used_gib"]))
    meta["idle_drift_gib"] = round(drift, 2)
    (outdir / "idle.txt").write_text(json.dumps(
        {"before": first_idle, "after": last_idle, "drift_gib": meta["idle_drift_gib"]}, indent=2))

    # ---- burn gate again.  A node that latched mid-session invalidates the session.
    for node in order[:need]:
        r = burn_gate(args, node, outdir, "after")
        if r["rc"] != 0:
            meta["burn_failed"] = True

    verdict = derive_verdict(cells, meta)
    (outdir / "results.json").write_text(json.dumps(
        {"cells": cells, "meta": meta, "verdict": verdict}, indent=2))
    (outdir / "fit-input.json").write_text(json.dumps(_fit_input(cells, verdict, meta), indent=2))
    (outdir / "README.md").write_text(_render_readme(args, cells, verdict, meta))

    print(json.dumps(verdict, indent=2))
    print(f"\nartifacts: {outdir}")
    return 0 if verdict.get("verdict") not in (None, "INVALID") else 2


def cmd_verdict(args):
    blob = json.loads(pathlib.Path(args.results).read_text())
    cells = blob["cells"] if isinstance(blob, dict) else blob
    meta = blob.get("meta", {}) if isinstance(blob, dict) else {}
    print(json.dumps(derive_verdict(cells, meta), indent=2))
    return 0


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("snapshot", help="one memory snapshot on this node")
    p.add_argument("--label", default="")
    p.set_defaults(fn=cmd_snapshot)

    p = sub.add_parser("idle", help="idle baseline + busy check; exit 3 if not idle")
    p.set_defaults(fn=cmd_idle)

    p = sub.add_parser("watch", help="poll meminfo and keep the worst moment")
    p.add_argument("--seconds", type=int, default=1800)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--stop-file", default="")
    p.set_defaults(fn=cmd_watch)

    for name, fn, helptext in (("plan", cmd_plan, "print the full cell plan, touch nothing"),
                               ("drive", cmd_drive, "run the sweep and write artifacts")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--arm", action="append", required=True, metavar="NAME:SLUG:NODES",
                       help="one arm, e.g. D2:qwen3-8b:2 (repeatable)")
        p.add_argument("--node-order", required=True,
                       type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                       help="node names in rank order, head first")
        p.add_argument("--gpu-util", type=float, default=0.70,
                       help="held FIXED in every cell; never swept (default 0.70)")
        p.add_argument("--repeats", type=int, default=MIN_REPEATS)
        p.add_argument("--lmds", default="lmds")
        p.add_argument("--deny", action="append", default=list(DEFAULT_DENY),
                       help="node names this script must never touch")
        if name == "drive":
            p.add_argument("--ssh-target", action="append", default=[], metavar="NAME=user@host",
                           help="raw ssh target per node (from `lmds node ls`)")
            p.add_argument("--out", default="docs/evidence/gpu-util-rank/")
            p.add_argument("--burn-cmd",
                           default="python3 scripts/hca_ab_bench.py burn",
                           help="GPU clock-latch gate, run per node; {node} is substituted. "
                                "Point this at `lmds burn` once the burn-gate work ships.")
            p.add_argument("--skip-burn", action="store_true",
                           help="run without the clock-latch gate and permanently taint the result")
            p.add_argument("--allow-busy", action="store_true",
                           help="measure a node that already has something resident (taints it)")
            p.add_argument("--start-timeout", type=int, default=2400)
            p.add_argument("--watch-seconds", type=int, default=2400)
            p.add_argument("--settle-seconds", type=int, default=45)
        p.set_defaults(fn=fn)

    p = sub.add_parser("verdict", help="re-derive the verdict from an existing results.json")
    p.add_argument("--results", required=True)
    p.set_defaults(fn=cmd_verdict)

    args = ap.parse_args()
    if args.cmd == "drive" and args.burn_cmd.strip() == "" and not args.skip_burn:
        raise SystemExit("ABORT: --burn-cmd is empty.  The clock-latch gate is not optional; "
                         "use --skip-burn if you really mean to taint the result.")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
