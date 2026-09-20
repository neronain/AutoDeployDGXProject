#!/usr/bin/env python3
"""
hca_ab_bench.py — A/B/C bandwidth test for NCCL_IB_HCA on DGX Spark (GB10).

Answers WS-1.2: does listing more than one RDMA function in NCCL_IB_HCA raise usable
fabric bandwidth between two Sparks, or does it only restate the same ~100 Gbps link
in the bus-bandwidth convention?  Procedure, thresholds and artifact layout:
docs/HCA-DUAL-TEST.md.

PROVENANCE
----------
The hypothesis under test, the NCCL_IB_HCA / NCCL_CROSS_NIC=1 pairing and the
23.6 GB/s busbw figure come from the third-party repository
"glm-5.3-uncensored-8x-dgx-spark" (launchers/launch-nvfp4-tp8.sh:82-85,
tools/run_nccl8.sh, tools/nccl_allreduce.py), which is MIT licensed:

    Copyright (c) 2026 im0xMagnus
    Licensed under the MIT License.

The all-reduce timing approach in rank_main() below (warm-up, then a timed loop of
fixed iteration count, converted to algbw and scaled to busbw by 2(n-1)/n) is derived
from that repo's tools/nccl_allreduce.py.  This file is an independent implementation:
no source text was copied.  It differs deliberately in three ways that matter to the
result -- the GID index is probed per node AND per HCA and never falls back to 3
(their run_nccl8.sh:15 hardcodes 3 and does not call their own gid probe), conditions
are interleaved A-B-C-A-B-C rather than measured once, and every cell's raw output and
environment are written to disk so the claim is auditable.

Full attribution belongs in THIRD-PARTY-LICENSES (WS-0.2).

This file is part of LMDS.  See the repository LICENSE.

USAGE
-----
    # end to end, from a machine with ssh to both nodes
    python3 scripts/hca_ab_bench.py drive \
        --node-a msi-1 --node-b msi-2 \
        --fabric-a 10.10.1.1 --fabric-b 10.10.1.2 \
        --repeats 3 --out docs/evidence/hca-dual/

    # single pieces, on a node
    python3 scripts/hca_ab_bench.py probe --fabric-ip 10.10.1.1
    python3 scripts/hca_ab_bench.py burn
    python3 scripts/hca_ab_bench.py rank      # needs RANK/WORLD_SIZE/MASTER_ADDR in env
"""

import argparse
import ipaddress
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time

IB = "/sys/class/infiniband"
SIZES_MIB = (8, 32, 128, 512, 1024)
DECISION_SIZE_MIB = 1024


# --------------------------------------------------------------------------- probe

def _read(path, default=""):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return default


def gid_suffix_for_ip(ip):
    """IPv4-mapped GID tail for `ip`, e.g. 10.10.1.1 -> '0a0a:0101'."""
    packed = ipaddress.IPv4Address(ip).packed
    return "%02x%02x:%02x%02x" % tuple(packed)


def resolve_gid_index(dev, netdev, fabric_ip):
    """Index of the RoCE v2 GID on `dev` that belongs to `netdev` and carries `fabric_ip`.

    Accepts an index only if all three hold: type is exactly 'RoCE v2', the GID's ndev
    is `netdev`, and the GID is the IPv4-mapped form of `fabric_ip`.  Raises if none
    match.  It never defaults to 3 -- a stale constant is how NCCL init hangs after a
    reboot (docs/UPGRADE-2026-09.md 1.1).
    """
    want = gid_suffix_for_ip(fabric_ip)
    base = f"{IB}/{dev}/ports/1"
    tried = []
    gid_dir = pathlib.Path(f"{base}/gids")
    for idx in sorted(int(p.name) for p in gid_dir.iterdir() if p.name.isdigit()):
        gid = _read(f"{base}/gids/{idx}")
        gtype = _read(f"{base}/gid_attrs/types/{idx}")
        gndev = _read(f"{base}/gid_attrs/ndevs/{idx}")
        if not gid or gid.endswith(":0000:0000"):
            continue
        tried.append(f"  [{idx}] gid={gid} type={gtype or '-'} ndev={gndev or '-'}")
        if gtype == "RoCE v2" and gndev == netdev and gid.endswith(want):
            return idx, tried
    raise SystemExit(
        f"FATAL: no RoCE v2 GID on {dev} ({netdev}) matching {fabric_ip} (expect tail {want}).\n"
        "Refusing to fall back to index 3. Candidates seen:\n" + "\n".join(tried or ["  (none)"])
    )


def discover(fabric_ip=None):
    """Every RDMA device with its PCI address, netdev, port state and physical port name."""
    out = []
    root = pathlib.Path(IB)
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        dev = d.name
        try:
            pci = os.path.basename(os.path.realpath(f"{IB}/{dev}/device"))
        except OSError:
            pci = ""
        netdevs = []
        try:
            netdevs = sorted(os.listdir(f"{IB}/{dev}/device/net"))
        except OSError:
            pass
        nd = netdevs[0] if netdevs else ""
        entry = {
            "device": dev,
            "pci": pci,
            "pci_domain": pci.split(":")[0] if pci else "",
            "netdev": nd,
            "phys_port_name": _read(f"/sys/class/net/{nd}/phys_port_name") if nd else "",
            "state": _read(f"{IB}/{dev}/ports/1/state"),
            "phys_state": _read(f"{IB}/{dev}/ports/1/phys_state"),
            "rate": _read(f"{IB}/{dev}/ports/1/rate"),
            "link_layer": _read(f"{IB}/{dev}/ports/1/link_layer"),
            "sys_image_guid": _read(f"{IB}/{dev}/sys_image_guid"),
            "node_guid": _read(f"{IB}/{dev}/node_guid"),
            "fw": _read(f"{IB}/{dev}/fw_ver"),
            "carrier": _read(f"/sys/class/net/{nd}/carrier") if nd else "",
            "speed": _read(f"/sys/class/net/{nd}/speed") if nd else "",
            "ipv4": _netdev_ipv4(nd) if nd else "",
        }
        entry["active"] = entry["state"].endswith("ACTIVE")
        out.append(entry)
    return out


def _netdev_ipv4(netdev):
    try:
        raw = subprocess.run(["ip", "-4", "-o", "addr", "show", "dev", netdev],
                             capture_output=True, text=True, timeout=10).stdout
        for line in raw.splitlines():
            parts = line.split()
            if "inet" in parts:
                return parts[parts.index("inet") + 1].split("/")[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def pick_roles(devs, fabric_ip):
    """Name HCA_A, HCA_XCAGE and HCA_SAMECAGE from live state, per docs/HCA-DUAL-TEST.md 4/P2.

    Deliberately observational: nothing here is hardcoded to rocep1s0f0 / roceP2p1s0f0,
    because on this fleet those two are function 0 of two DIFFERENT PCI domains and are
    administratively disabled on at least one cabled pair.
    """
    primary = next((d for d in devs if d["ipv4"] == fabric_ip), None)
    if primary is None:
        raise SystemExit(
            f"FATAL: no RDMA device carries fabric IP {fabric_ip}.\n"
            "Devices seen:\n" + "\n".join(
                f"  {d['device']:<14} {d['pci']:<13} {d['netdev']:<15} "
                f"ip={d['ipv4'] or '-':<15} {d['state']}" for d in devs))
    if not primary["active"]:
        raise SystemExit(f"FATAL: {primary['device']} is not ACTIVE ({primary['state']}).")

    def best(cands):
        """Prefer a device with a routable IPv4 -- a link-local-only port has a RoCE v2
        GID but no configured fabric path, so NCCL's choice there is not the thing we
        mean to measure.  Fall back to an ACTIVE link-local port and flag it."""
        cands = [d for d in cands if d["active"]]
        routable = [d for d in cands if d["ipv4"] and not d["ipv4"].startswith("169.254.")]
        if routable:
            return routable[0]
        for d in cands:
            d["note"] = "link-local only (169.254.x) -- no configured fabric subnet on this port"
        return cands[0] if cands else None

    same_cage = best([d for d in devs
                      if d["pci_domain"] == primary["pci_domain"]
                      and d["device"] != primary["device"]])
    x_cage = best([d for d in devs if d["pci_domain"] != primary["pci_domain"]])
    return {
        "HCA_A": primary,
        "HCA_XCAGE": x_cage,
        "HCA_SAMECAGE": same_cage,
    }


def cmd_probe(args):
    devs = discover()
    report = {"host": os.uname().nodename, "devices": devs}
    if args.fabric_ip:
        roles = pick_roles(devs, args.fabric_ip)
        report["roles"] = {k: (v["device"] if v else None) for k, v in roles.items()}
        report["gid"] = {}
        for role, dev in roles.items():
            if dev is None:
                continue
            ip = dev["ipv4"] or args.fabric_ip
            if not dev["ipv4"]:
                report["gid"][role] = {"device": dev["device"], "index": None,
                                       "note": "no IPv4 on this netdev; not usable as an NCCL HCA"}
                continue
            idx, tried = resolve_gid_index(dev["device"], dev["netdev"], ip)
            report["gid"][role] = {"device": dev["device"], "netdev": dev["netdev"],
                                   "ip": ip, "index": idx, "candidates": tried,
                                   "note": dev.get("note", "")}
    print(json.dumps(report, indent=2))
    return 0


# ---------------------------------------------------------------------------- burn

BURN_SECONDS = 15
BURN_SAMPLE_AT = 12
BURN_FAIL_TFLOPS = 50.0


def cmd_burn(args):
    """GPU clock latch gate (docs/UPGRADE-2026-09.md 1.4).

    A latched GB10 runs 631-949 MHz under load and nvidia-smi reports nothing wrong.
    In a collective every rank waits for the slowest, so one latched node caps the whole
    measurement and reads as 'the HCA change did nothing'.
    """
    import torch
    torch.cuda.set_device(0)
    n = 4096
    a = torch.randn(n, n, dtype=torch.float16, device="cuda")
    b = torch.randn(n, n, dtype=torch.float16, device="cuda")
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()

    flop_per_matmul = 2.0 * n ** 3
    sample = None
    iters = 0
    t0 = time.perf_counter()
    while True:
        a @ b
        iters += 1
        if iters % 20 == 0:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if sample is None and elapsed >= BURN_SAMPLE_AT:
                sample = _smi()
            if elapsed >= BURN_SECONDS:
                break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tflops = iters * flop_per_matmul / elapsed / 1e12
    sample = sample or _smi()
    verdict = "PASS" if tflops >= BURN_FAIL_TFLOPS else "FAIL"
    print(json.dumps({
        "host": os.uname().nodename, "tflops": round(tflops, 1),
        "clocks_sm_mhz": sample.get("clocks.sm"), "power_w": sample.get("power.draw"),
        "seconds": round(elapsed, 1), "iters": iters,
        "fail_below_tflops": BURN_FAIL_TFLOPS, "verdict": verdict,
        "remedy": None if verdict == "PASS" else
                  "EC clock latch: power off and UNPLUG the adapter 30-60s. "
                  "A normal reboot does not clear it. Re-run this gate after.",
    }, indent=2))
    return 0 if verdict == "PASS" else 2


def _smi():
    try:
        raw = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        sm, pw = (x.strip() for x in raw.split(","))
        return {"clocks.sm": float(sm), "power.draw": float(pw)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


# ---------------------------------------------------------------------------- rank

def cmd_rank(args):
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    # ring all-reduce: each byte crosses the wire 2(n-1)/n times.  At n=2 this is 1.0,
    # so busbw == algbw and there is no factor for a headline number to hide behind.
    factor = 2.0 * (world - 1) / world
    rows = []

    for mib in SIZES_MIB:
        elems = mib * 1024 * 1024 // 4
        buf = torch.ones(elems, dtype=torch.float32, device="cuda")
        for _ in range(5):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()

        iters = 20 if mib <= 128 else 8
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters

        nbytes = elems * 4
        algbw = nbytes / dt / 1e9
        rows.append({"size_mib": mib, "ms": round(dt * 1e3, 3),
                     "algbw_gbs": round(algbw, 3),
                     "busbw_gbs": round(algbw * factor, 3)})
        del buf
        torch.cuda.empty_cache()

    check = torch.ones(4096, device="cuda")
    dist.all_reduce(check)
    correct = bool((check == world).all().item())

    if rank == 0:
        print("HCA_AB_RESULT " + json.dumps({
            "world": world, "busbw_factor": factor, "correctness": correct,
            "torch": torch.__version__, "nccl": ".".join(map(str, torch.cuda.nccl.version())),
            "rows": rows,
        }))
    dist.destroy_process_group()
    return 0 if correct else 3


# --------------------------------------------------------------------------- drive

CONDITIONS = ("A", "B", "C")


def _ssh(host, argv, env=None, capture=True, source=None):
    """Run this script on `host` by piping its source to python3 -- installs nothing."""
    pre = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (env or {}).items())
    remote = f"{pre} python3 - {' '.join(shlex.quote(a) for a in argv)}".strip()
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, remote]
    return subprocess.Popen(cmd,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT if capture else None,
                            text=True)


def _ssh_run(host, argv, env=None, source=""):
    p = _ssh(host, argv, env)
    out, _ = p.communicate(source)
    return p.returncode, out


def _json_tail(text, marker=None):
    if marker:
        for line in text.splitlines():
            if line.startswith(marker):
                return json.loads(line[len(marker):])
        return None
    start = text.find("{")
    return json.loads(text[start:]) if start >= 0 else None


def cmd_drive(args):
    src = pathlib.Path(__file__).read_text()
    stamp = time.strftime("%Y-%m-%d", time.gmtime())
    outdir = pathlib.Path(args.out) / f"{stamp}-{args.node_a}-{args.node_b}"
    raw = outdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (outdir / "args.json").write_text(json.dumps({"argv": sys.argv, "utc": stamp}, indent=2))

    nodes = [(args.node_a, args.fabric_a), (args.node_b, args.fabric_b)]

    # ---- P1 burn gate, both nodes, before the sweep
    burn_before = {}
    for host, _ in nodes:
        rc, out = _ssh_run(host, ["burn"], source=src)
        burn_before[host] = out
        print(f"[burn:{host}] rc={rc}\n{out}")
        if rc != 0 and not args.skip_burn:
            (outdir / "burn.txt").write_text(json.dumps(burn_before, indent=2))
            raise SystemExit(
                f"ABORT: {host} failed the GPU burn gate. A latched node caps every rank.\n"
                "Fix it (power off, unplug adapter 30-60s) and re-run. "
                "--skip-burn overrides, and taints the result.")

    # ---- P2/P4 fabric + GID, both nodes
    probes = {}
    for host, fip in nodes:
        rc, out = _ssh_run(host, ["probe", "--fabric-ip", fip], source=src)
        if rc != 0:
            print(out, file=sys.stderr)
            raise SystemExit(f"ABORT: probe failed on {host}.")
        probes[host] = _json_tail(out)
    (outdir / "fabric.txt").write_text(json.dumps(probes, indent=2))
    (outdir / "gid-probe.txt").write_text(json.dumps(
        {h: p.get("gid") for h, p in probes.items()}, indent=2))

    runnable = [c for c in CONDITIONS if _condition_available(c, probes, nodes)]
    print(f"conditions runnable on this pair: {', '.join(runnable)}")
    if "A" not in runnable or "B" not in runnable:
        raise SystemExit("ABORT: need at least conditions A and B to say anything.")

    # ---- interleaved sweep: A B C - A B C - A B C  (never A A A - B B B)
    cells = []
    for rep in range(1, args.repeats + 1):
        for cond in runnable:
            print(f"--- repeat {rep} condition {cond} ---")
            cell = _run_cell(cond, rep, nodes, probes, src, raw, args)
            cells.append(cell)
            (outdir / "results.json").write_text(json.dumps(cells, indent=2))

    _write_env(outdir, cells)
    summary = _summarise(cells, runnable)
    (outdir / "README.md").write_text(_render_readme(args, summary, cells, runnable, probes))
    print(json.dumps(summary, indent=2))
    print(f"\nartifacts: {outdir}")
    return 0


def _condition_available(cond, probes, nodes):
    role = {"A": "HCA_A", "B": "HCA_XCAGE", "C": "HCA_SAMECAGE"}[cond]
    return all(probes[h]["roles"].get(role) for h, _ in nodes)


def _cell_env(cond, host, probe):
    """The ONLY three variables that differ between conditions (docs/HCA-DUAL-TEST.md 5)."""
    gid = probe["gid"]
    a_dev, a_idx = gid["HCA_A"]["device"], gid["HCA_A"]["index"]
    if cond == "A":
        return {"NCCL_IB_HCA": a_dev, "NCCL_CROSS_NIC": "0",
                "NCCL_IB_GID_INDEX": str(a_idx)}
    role = "HCA_XCAGE" if cond == "B" else "HCA_SAMECAGE"
    return {"NCCL_IB_HCA": f"{a_dev},{gid[role]['device']}", "NCCL_CROSS_NIC": "1",
            "NCCL_IB_GID_INDEX": f"{a_idx},{gid[role]['index']}"}


def _run_cell(cond, rep, nodes, probes, src, raw, args):
    (host_a, fab_a), (host_b, _) = nodes
    procs, logs = [], {}
    for rank, (host, _) in enumerate(nodes):
        probe = probes[host]
        env = {
            "RANK": rank, "WORLD_SIZE": 2,
            "MASTER_ADDR": fab_a, "MASTER_PORT": str(args.master_port),
            "NCCL_NET": "IB", "NCCL_IB_DISABLE": "0",
            "NCCL_IB_ROCE_VERSION_NUM": "2", "NCCL_IB_ADDR_FAMILY": "AF_INET",
            "NCCL_SOCKET_IFNAME": _netdev_of(probe, "HCA_A"),
            "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET,ENV",
        }
        env.update(_cell_env(cond, host, probe))
        logs[rank] = env
        procs.append((rank, host, env, _ssh(host, ["rank"], env)))

    results, raw_out = {}, {}
    for rank, host, env, p in procs:
        out, _ = p.communicate(src, timeout=args.timeout)
        raw_out[rank] = out
        (raw / f"{cond}-rep{rep}-rank{rank}.log").write_text(
            "# host=%s\n# env=%s\n%s" % (host, json.dumps(env), out))
        if rank == 0:
            results = _json_tail(out, "HCA_AB_RESULT ") or {}

    # Did NCCL really open every HCA we listed?  A silent fallback to one device is the
    # most likely way B looks like A, and it is invisible without this.
    listed = logs[0]["NCCL_IB_HCA"].split(",")
    seen = [d for d in listed if d in raw_out.get(0, "")]
    return {
        "condition": cond, "repeat": rep,
        "env": {k: logs[0][k] for k in ("NCCL_IB_HCA", "NCCL_CROSS_NIC", "NCCL_IB_GID_INDEX")},
        "hcas_listed": listed, "hcas_seen_in_log": seen,
        "all_hcas_used": len(seen) == len(listed),
        "correctness": results.get("correctness"),
        "rows": results.get("rows", []),
        "decision_algbw_gbs": next(
            (r["algbw_gbs"] for r in results.get("rows", [])
             if r["size_mib"] == DECISION_SIZE_MIB), None),
    }


def _netdev_of(probe, role):
    dev = probe["roles"][role]
    return next(d["netdev"] for d in probe["devices"] if d["device"] == dev)


def _write_env(outdir, cells):
    lines = []
    for c in cells:
        lines.append(f"[{c['condition']} rep{c['repeat']}] " +
                     " ".join(f"{k}={v}" for k, v in c["env"].items()))
    (outdir / "env.txt").write_text("\n".join(lines) + "\n")


def _summarise(cells, runnable):
    """Median/min/max per condition, the A-B-A drift check, and the pass rule."""
    def counted(cond):
        return [c for c in cells
                if c["condition"] == cond and c["correctness"] and c["decision_algbw_gbs"]]

    def med(xs):
        s = sorted(xs)
        n = len(s)
        return None if not n else (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)

    stats = {}
    for cond in runnable:
        vals = [c["decision_algbw_gbs"] for c in counted(cond)]
        if not vals:
            stats[cond] = {"n": 0}
            continue
        m = med(vals)
        stats[cond] = {"n": len(vals), "median": round(m, 3),
                       "min": round(min(vals), 3), "max": round(max(vals), 3),
                       "spread_frac": round((max(vals) - min(vals)) / m, 3),
                       "all_hcas_used": all(c["all_hcas_used"] for c in counted(cond))}

    a_vals = [c["decision_algbw_gbs"] for c in counted("A")]
    drift = None
    if len(a_vals) >= 2 and stats["A"].get("median"):
        drift = round(abs(a_vals[0] - a_vals[-1]) / stats["A"]["median"], 3)

    out = {"decision_size_mib": DECISION_SIZE_MIB, "metric": "algbw_gbs",
           "per_condition": stats, "aba_drift_frac": drift,
           "aba_ok": (drift is not None and drift <= 0.10)}

    for cond in [c for c in runnable if c != "A"]:
        out[f"verdict_{cond}_vs_A"] = _verdict(stats, cond, out["aba_ok"])
    return out


def _verdict(stats, cond, aba_ok):
    a, b = stats.get("A", {}), stats.get(cond, {})
    if not a.get("n") or not b.get("n"):
        return "INCONCLUSIVE (no counted cells)"
    if not aba_ok:
        return "INCONCLUSIVE (A-B-A drift >10%: session drifted, re-run)"
    ratio = b["median"] / a["median"]
    if b["spread_frac"] > abs(ratio - 1):
        return f"NULL (within-condition spread {b['spread_frac']:.2f} exceeds the delta)"
    if not b.get("all_hcas_used"):
        return "UNTESTED (NCCL did not open every listed HCA -- silent fallback)"
    if ratio >= 1.5 and b["min"] > a["max"]:
        return f"PASS ({ratio:.2f}x)"
    if ratio < 1.2:
        return f"FAIL ({ratio:.2f}x -- effect measurably absent)"
    return f"INCONCLUSIVE ({ratio:.2f}x)"


def _render_readme(args, summary, cells, runnable, probes):
    v = "\n".join(f"- **{c} vs A:** {summary.get(f'verdict_{c}_vs_A')}"
                  for c in runnable if c != "A")
    return f"""# WS-1.2 HCA dual-function A/B — {args.node_a} <-> {args.node_b}

Procedure: `docs/HCA-DUAL-TEST.md`. Metric: median **algbw** at {DECISION_SIZE_MIB} MiB, n=2
(at n=2 the busbw factor is 1.0, so busbw == algbw).

{v}

A-B-A drift: {summary.get('aba_drift_frac')} (ok if <= 0.10) — {summary.get('aba_ok')}

| condition | n | median GB/s | min | max | spread | all HCAs used |
|---|---|---|---|---|---|---|
""" + "\n".join(
        "| {c} | {n} | {med} | {mn} | {mx} | {sp} | {hc} |".format(
            c=c, n=s.get("n"), med=s.get("median"), mn=s.get("min"), mx=s.get("max"),
            sp=s.get("spread_frac"), hc=s.get("all_hcas_used"))
        for c, s in summary["per_condition"].items()) + f"""

Raw NCCL_DEBUG=INFO stdout for every cell, including discarded ones, is in `raw/`.
Environments as passed: `env.txt`. GID indices and why they were chosen: `gid-probe.txt`.
Fabric inventory of both nodes: `fabric.txt`. Burn gate: `burn.txt`.

Compare against the converted third-party figure: 23.6 GB/s busbw at n=8 is
23.6 / 1.75 = **13.5 GB/s algbw** ≈ 108 Gbps per node. A condition-B median near that
reproduces their number while showing it is the ~100 Gbps PCIe Gen5 x4 ceiling
(`docs/RUNBOOK-MULTI-NODE.md:330`), not an escape from it.

Cells recorded: {len(cells)}. Conditions runnable on this pair: {', '.join(runnable)}.
"""


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="fabric inventory + per-HCA GID resolution")
    p.add_argument("--fabric-ip", help="this node's head<->worker fabric IP")
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("burn", help="GPU clock latch gate (UPGRADE-2026-09 1.4)")
    p.set_defaults(fn=cmd_burn)

    p = sub.add_parser("rank", help="one rank of the all-reduce (needs RANK/WORLD_SIZE/MASTER_ADDR)")
    p.set_defaults(fn=cmd_rank)

    p = sub.add_parser("drive", help="full interleaved A/B/C sweep over ssh, with artifacts")
    p.add_argument("--node-a", required=True, help="ssh host of rank 0")
    p.add_argument("--node-b", required=True, help="ssh host of rank 1")
    p.add_argument("--fabric-a", required=True, help="rank 0 fabric IP (also MASTER_ADDR)")
    p.add_argument("--fabric-b", required=True, help="rank 1 fabric IP")
    p.add_argument("--repeats", type=int, default=3, help="repeats per condition (>=3)")
    p.add_argument("--out", default="docs/evidence/hca-dual/")
    p.add_argument("--master-port", type=int, default=29500)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--skip-burn", action="store_true",
                   help="proceed despite a failed burn gate; taints the result")
    p.set_defaults(fn=cmd_drive)

    args = ap.parse_args()
    if args.cmd == "drive" and args.repeats < 3:
        print("WARNING: <3 repeats cannot satisfy the A-B-A check in docs/HCA-DUAL-TEST.md 7.",
              file=sys.stderr)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
