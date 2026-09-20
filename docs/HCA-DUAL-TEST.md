# HCA-DUAL-TEST — does listing both RDMA functions in `NCCL_IB_HCA` actually double fabric bandwidth?

**Work stream:** WS-1.2 (`docs/UPGRADE-2026-09.md` §1.2)
**Status:** procedure ready · **not yet run** · no artifact exists · nothing here may be cited as `hardware-validated`
**Companion to:** `docs/RUNBOOK-MULTI-NODE.md:330` (the ~100 Gbps / PCIe Gen5 x4 ceiling) and §1.1 (GID probe), §1.4 (GPU clock latch)

This document is a runnable A/B plan for whoever has two DGX Spark (GB10) machines cabled to each
other. Read §1 and §2 before running anything — the recon below changed what the experiment has to be.

---

## 1 · Finding 0 — the third-party premise does not describe our hardware

WS-1.2 records the third-party claim as: *`rocep1s0f0` and `roceP2p1s0f0` are two PCIe functions of
the **same** QSFP cage; list both and you escape the ~100G per-function ceiling.*

Read-only recon across our fleet on 2026-09-20 contradicts the "same cage" part. Every GB10 node that
has ConnectX-7 exposes **four** RDMA devices on **two separate PCI domains**, two functions each:

| RDMA device | PCI address | netdev | `phys_port_name` |
|---|---|---|---|
| `rocep1s0f0`   | `0000:01:00.0` | `enp1s0f0np0`   | `p0` |
| `rocep1s0f1`   | `0000:01:00.1` | `enp1s0f1np1`   | `p1` |
| `roceP2p1s0f0` | `0002:01:00.0` | `enP2p1s0f0np0` | `p0` |
| `roceP2p1s0f1` | `0002:01:00.1` | `enP2p1s0f1np1` | `p1` |

All four report one `sys_image_guid` and one `switchid` — a single integrated ConnectX-7 ASIC — but
they are presented on two different PCI root complexes. The `p0`/`p1` pair sits **inside** a PCI
domain; the two domains are the two physical QSFP cages, exactly as
`docs/RUNBOOK-MULTI-NODE.md` §8 already describes them.

So `rocep1s0f0` + `roceP2p1s0f0` is **`p0` of cage 1 plus `p0` of cage 2** — one port from each of two
different cages, not two functions of one cage. Two consequences:

1. **Their env string is not portable to our nodes.** On our cabled pair `msi-1` ↔ `msi-2` the two
   `f0` devices are `phys_state DISABLED` with `rx_bytes=0`; the traffic-bearing devices are
   `rocep1s0f1` and `roceP2p1s0f1`. Pasting `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0` there selects two
   dead ports. **Resolve device names per node from live state. Never hardcode them** — the same
   rule §1.1 sets for `NCCL_IB_GID_INDEX`, for the same reason.
2. **The "two functions of one cage" configuration is still worth testing** — it is just a different
   pair than they named. On our hardware that pair is `rocep1s0f0` + `rocep1s0f1` (both in domain
   `0000`). It is condition **C** below.

---

## 2 · Finding 1 — the unit trap: their 23.6 GB/s may already *be* our ~100 Gbps

`busbw = algbw × 2(n−1)/n` for a ring all-reduce. The third-party figure of **23.6 GB/s busbw** was
taken on an **8-node** cluster, where that factor is `2×7/8 = 1.75`. Converting back:

```
algbw = 23.6 / 1.75 = 13.5 GB/s  ≈  108 Gbps per node on the wire
```

That is the same order as the `~100 Gbps per link` ceiling our own runbook documents. **Their headline
number is not obviously evidence of escaping that ceiling** — it may simply be that ceiling restated
in the bus-bandwidth convention, which multiplies a per-link rate up by 1.75 at n=8.

This is the cross-repo unit mismatch that `docs/UPGRADE-2026-09.md` §"คุณภาพหลักฐาน" point 1 warns
about, in a new place. Therefore:

- **Record and compare `algbw`, not `busbw`,** when asking "did the wire get faster?"
- Report both, but make `algbw` the decision variable.
- At **n=2** the factor is `2×1/2 = 1.0`, so `busbw == algbw`. A 2-node test is *cleaner* than their
  8-node test for this question — there is no factor to hide behind.

Reference points for a 2-node run, per link:

| | GB/s | Gbps |
|---|---|---|
| 200G (2X NDR) wire rate | 25.0 | 200 |
| PCIe Gen5 x4 usable (the suspected condition-A ceiling) | ~12.5 | ~100 |
| Their claim, converted to per-node algbw | ~13.5 | ~108 |

If condition A lands near 12.5 GB/s and condition B lands near 23–24 GB/s, the hypothesis is real.
If B lands near 13.5 GB/s, we have reproduced their number and shown it is **not** a doubling.

---

## 3 · Hardware that can run this

Recon of 2026-09-20 (read-only, see §10 for what was run). Of eleven reachable machines, **five** have
ConnectX-7 at all, and only **two pairs** are cabled to each other:

| Pair | Link state | Condition A/B runnable | Condition C runnable |
|---|---|---|---|
| `msi-1` ↔ `msi-2` | `rocep1s0f1`+`roceP2p1s0f1` ACTIVE 200G on `10.10.1.0/24` + `10.10.2.0/24`; both `f0` DISABLED | **yes** | no — would require bringing `f0` up (a config change) |
| `gigabyte01` ↔ `gigabyte02` | **all four** devices ACTIVE 200G; `f1` pair configured on `10.100.152.0/24` + `10.100.153.0/24`; `f0` pair link-up with APIPA only | **yes** | **yes**, with a caveat — the only pair where C needs no config change, but it runs over link-local addresses (see below) |

On `gigabyte01` the role picker resolves, against live sysfs (run 2026-09-20, read-only):

```
HCA_A        rocep1s0f1    enp1s0f1np1    10.100.152.1    GID_INDEX=3
HCA_XCAGE    roceP2p1s0f1  enP2p1s0f1np1  10.100.153.1    GID_INDEX=3
HCA_SAMECAGE rocep1s0f0    enp1s0f0np0    169.254.21.127  GID_INDEX=3   link-local only
```

Two things to read out of that. **The index 3 here was derived, not assumed** — it was accepted
because the GID is `RoCE v2`, its ndev matches, and it is the IPv4-mapped form of that port's own
address. That it coincides with the hardcoded constant on this node is exactly why the constant
survives until the day it does not. And **condition C on this pair runs over a link-local-addressed
port**: `rocep1s0f0` is link-up at 200G but carries only APIPA, so P3's both-ends ping is not optional
there — it is the only thing that establishes the two `f0` ports are actually on one wire.

`msi-5` has all four ConnectX devices but **no cable** (all four DISABLED, no fabric IP). The four
`spark-01..04` and `dgx-msi` boxes are GB10 but expose **no ConnectX at all** — `/sys/class/infiniband/`
is empty and `lspci` shows only the onboard Realtek 1GbE. They cannot take part.

> **Before you use either pair: they are live.** `gigabyte01`↔`gigabyte02` shows ~680 MB already moved
> on `enp1s0f1np1`. Confirm with the owner that the pair is idle, and check `nvidia-smi` and
> `docker ps` on both nodes, before running a benchmark that saturates their fabric.

---

## 4 · Preconditions — all five must pass, in order, and be captured as files

Skipping any of these makes the measurement uninterpretable rather than merely noisy.

### P0 · Both nodes idle
No vLLM / Ray / training container holding GPUs or the fabric.
```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
docker ps
```
Record both. Any resident process → stop, or pick another pair.

### P1 · GPU burn gate — **the one that silently ruins everything**
`docs/UPGRADE-2026-09.md` §1.4: the embedded controller can latch a GB10 at 631–949 MHz, and
`nvidia-smi` reports nothing wrong — P0, persistence on, application clocks 2418, no clock event
reason, clean kernel log. In a collective every rank waits for the slowest, so **one latched node
caps the whole measurement** and you will read it as "the HCA change did nothing".

Run on **both** nodes: fp16 4096² matmul for 15 s, sample `clocks.sm` and `power.draw` at t=12 s.

| | healthy | latched |
|---|---|---|
| `clocks.sm` | 2.2–2.4 GHz | ~700–950 MHz |
| `power.draw` | ≥80 W | <20 W |
| achieved | 75–90 TFLOPS | — |
| **fail threshold** | **<50 TFLOPS** | |

A latched node is fixed only by powering off and **unplugging the adapter for 30–60 s** — a normal
reboot does not clear it, because the EC keeps standby power while the plug is in. Re-run the gate
after. **Do not start the A/B sweep until both nodes pass.**

### P2 · Fabric inventory, captured not assumed
On both nodes:
```bash
ls /sys/class/infiniband/
rdma link show
for d in /sys/class/infiniband/*; do
  echo "$(basename $d) pci=$(basename $(readlink -f $d/device)) \
state=$(cat $d/ports/1/state) rate=$(cat $d/ports/1/rate) \
ll=$(cat $d/ports/1/link_layer) sys=$(cat $d/sys_image_guid)"
done
ip -4 -br addr
for i in enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1; do
  [ -d /sys/class/net/$i ] && echo "$i portname=$(cat /sys/class/net/$i/phys_port_name) \
carrier=$(cat /sys/class/net/$i/carrier) speed=$(cat /sys/class/net/$i/speed)"
done
```
From this, name the three devices the conditions need — **by observation, on each node separately**:

- `HCA_A` — the device whose netdev holds the head↔worker fabric IP. Condition A uses only this.
- `HCA_XCAGE` — the ACTIVE device in the **other** PCI domain. Condition B adds this.
- `HCA_SAMECAGE` — the other function in the **same** PCI domain as `HCA_A`. Condition C adds this.

Every device used must read `state=4: ACTIVE` and `rate=200 Gb/sec`. A device at
`phys_state DISABLED` is not a test condition, it is a typo.

### P3 · Same subnet, both directions
Each link pair must be a real L2 path, verified from both ends, not inferred from carrier:
```bash
ping -c3 -I <local-fabric-ip> <peer-fabric-ip>
```
`carrier=1` only means "a cable is in this socket" — it does not say who is on the far end.

### P4 · GID index resolved per node **and per HCA**, never defaulted
`docs/UPGRADE-2026-09.md` §1.1 and `docs/DGX-SPARK-VLLM-FIELD-NOTES.md:111`: a constant drifts after
reboot and NCCL init then hangs. Our own
`src/lmds/generator/templates/stacked-vllm-controller.sh.j2:126` still carries
`NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"`, and the third-party validation harness
(`tools/run_nccl8.sh:15`) hardcodes `NCCL_IB_GID_INDEX=3` — *their own gid probe tool exists and that
script does not call it.* **Do not reproduce that mistake here.**

Accept an index `i` for device `D` only if **all three** hold:
1. `gid_attrs/types/i` is exactly `RoCE v2`;
2. `gid_attrs/ndevs/i` is the netdev of `D`;
3. `gids/i` is the IPv4-mapped form of that netdev's own fabric IP
   (`0000:0000:0000:0000:0000:ffff:<ip as hex>`).

If no index matches — **abort loudly. Do not fall back to 3.** Condition B uses two HCAs whose indices
may differ; resolve one per HCA and pass them in the order matching `NCCL_IB_HCA`.

### P5 · Identical software on both nodes
Same container image or venv, same torch, same NCCL. Record
`torch.__version__`, `torch.cuda.nccl.version()`, `nvidia-smi --query-gpu=driver_version`, kernel, and
ConnectX firmware (`ethtool -i <iface> | grep firmware`) on both. Our fleet is **not** uniform —
observed kernels `6.17.0-1014/1026/1029/1032-nvidia` and drivers `580.142` / `580.159.03` /
`580.173.02`. A mismatched pair is a confound, and must at minimum be recorded.

---

## 5 · The conditions

Two nodes, one GPU per node, `WORLD_SIZE=2`. **Everything outside the three lines below is byte-identical
across conditions** — same image, same message sizes, same iteration counts, same channel settings,
same socket interface.

Fixed in all conditions:
```bash
NCCL_NET=IB
NCCL_IB_DISABLE=0
NCCL_IB_ROCE_VERSION_NUM=2
NCCL_IB_ADDR_FAMILY=AF_INET
NCCL_SOCKET_IFNAME=<netdev of HCA_A>      # bootstrap only, same in A, B and C
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,NET,ENV            # so the log proves which HCAs were really used
```

| | `NCCL_IB_HCA` | `NCCL_CROSS_NIC` | `NCCL_IB_GID_INDEX` |
|---|---|---|---|
| **A** baseline, one HCA | `$HCA_A` | `0` | `$GID_A` |
| **B** cross-cage pair (the WS-1.2 claim, corrected to live device names) | `$HCA_A,$HCA_XCAGE` | `1` | `$GID_A,$GID_XCAGE` |
| **C** same-cage pair (what "two functions of one cage" actually means here) | `$HCA_A,$HCA_SAMECAGE` | `1` | `$GID_A,$GID_SAMECAGE` |

Condition **C** is an addition to the WS-1.2 brief, justified by §1: the pair the third-party repo
named is a cross-cage pair on our hardware, so their stated *mechanism* is only tested by C. Run C
wherever it is runnable (`gigabyte01`↔`gigabyte02` today). If a pair cannot run C, record A and B and
say so — do not bring a disabled port up just to complete the table; that is a config change and a
separate, owner-approved action.

> `NCCL_CROSS_NIC=1` is deliberately tied to the multi-HCA conditions because that is how the claim is
> stated. It therefore co-varies with the HCA count, and a positive result cannot attribute the gain
> to one or the other. If B or C passes, **the follow-up run is B with `NCCL_CROSS_NIC=0`** to separate
> them. Note this in the result rather than claiming the HCA list caused it.

---

## 6 · What to run

Use `scripts/hca_ab_bench.py` (in this repo — see its header for provenance). It performs the GID
probe of P4, refuses to run on an unresolved index, sweeps the conditions in the order of §7, and
writes the artifact tree of §8.

```bash
# from a machine that can ssh to both nodes
python3 scripts/hca_ab_bench.py drive \
  --node-a msi-1 --node-b msi-2 \
  --fabric-a 10.10.1.1 --fabric-b 10.10.1.2 \
  --repeats 3 \
  --out docs/evidence/hca-dual/
```

Message sizes: 8, 32, 128, 512, 1024 MiB. 5 warm-up iterations discarded, then 20 timed iterations at
≤128 MiB and 8 at >128 MiB, with a `torch.cuda.synchronize()` and a `dist.barrier()` before the timed
loop. **The decision size is 1024 MiB** — small messages are latency-bound and will show no HCA effect
whatever the truth is.

A correctness check runs after each cell: all-reduce a buffer of ones; every element must equal
`WORLD_SIZE`. **A cell that fails correctness is discarded, not averaged in.**

If you would rather drive it by hand, the only thing that matters is that the three env lines of §5
are the *only* difference between cells and that raw stdout is kept.

`ib_write_bw` is present on the fleet and is useful as a per-device sanity floor, but it exercises one
device at a time and **cannot answer this question** — the multi-HCA path lives in NCCL, not in
perftest. Do not substitute it.

---

## 7 · Repeat discipline — non-negotiable

The repos this hypothesis comes from measured **n=1 per cell**, and the project's own evidence review
records **~33.7% run-to-run spread on prefill** and GB10 slow-state shifting a single cell by ~1.5×.
A 2× effect measured once against a distribution that wide is not a result.

- **≥3 repeats per condition.**
- **Interleave, never block.** Run `A B C · A B C · A B C`. Do **not** run `A A A · B B B` — a node
  drifting, warming, or latching mid-session then shows up as a condition effect.
- The **A-B-A property** is what the interleaving buys: the first and last `A` must agree. If
  `|A_first − A_last| / median(A) > 0.10`, the session drifted. **Discard the session and re-run**;
  do not report it.
- Report **median, min and max** for every cell. Never a bare mean.
- Report the within-condition spread `(max−min)/median` alongside the between-condition delta. If the
  within-condition spread is larger than the A→B delta, the result is **NULL**, whatever the medians say.

---

## 8 · Pass / fail

Decision variable: **median `algbw` at 1024 MiB, n=2**. (At n=2, `busbw == algbw`; record both anyway.)

| Outcome | Rule |
|---|---|
| **PASS** | `median(B) ≥ 1.5 × median(A)` **and** `min(B) > max(A)` **and** all three A-cells within 10% of each other **and** correctness PASS in every counted cell |
| **FAIL** | `median(B) < 1.2 × median(A)` with the A-B-A drift check satisfied — i.e. the effect is measurably absent, not merely unproven |
| **INCONCLUSIVE** | anything else: overlapping ranges, a 1.2–1.5× median ratio, a failed A-B-A check, or within-condition spread exceeding the delta |

Same rule applied independently to C vs A.

Secondary observations to record regardless of outcome:
- **Did NCCL actually use both HCAs?** Grep the `NCCL_DEBUG=INFO` log for the `NET/IB` init lines
  naming each device. A silent fallback to one HCA is the most likely way B looks like A, and it is
  invisible without this. **A "FAIL" without this grep is not a FAIL, it is an untested cell.**
- Whether B's algbw lands near **13.5 GB/s** — the converted third-party figure of §2. Landing there
  reproduces their number while showing it is the ~100G ceiling, not an escape from it. That is a
  publishable result and the most likely one.

---

## 9 · Artifacts — where raw output must go

Project rule: **a claim without an artifact is not validated.** `docs/UPGRADE-2026-09.md` grades the
source of this very hypothesis as *"คำกล่าวอ้าง"* (a claim) precisely because its repo has no
`results/` folder — every number lives in a README with no log behind it. We do not get to repeat that.

```
docs/evidence/hca-dual/<UTC-yyyy-mm-dd>-<nodeA>-<nodeB>/
├── README.md            # outcome, one paragraph, links each number to a file below
├── results.json         # every cell: condition, repeat index, size, algbw, busbw, correctness
├── env.txt              # full env of each cell, as passed — the audit trail for "only 3 vars changed"
├── args.json            # real argv of the driver
├── fabric.txt           # P2 output from both nodes
├── gid-probe.txt        # P4 output: index chosen per HCA per node, and why
├── burn.txt             # P1 clocks/power/TFLOPS per node, before and after the sweep
├── versions.txt         # P5: torch, NCCL, driver, kernel, ConnectX firmware, both nodes
└── raw/
    ├── A-rep1-rank0.log  A-rep1-rank1.log
    ├── B-rep1-rank0.log  B-rep1-rank1.log
    └── ...               # unfiltered NCCL_DEBUG=INFO stdout for every cell, including discarded ones
```

Keep the discarded cells. A cell dropped for failed correctness or an A-B-A violation is evidence
about the fleet and must stay in `raw/` with the reason recorded in `README.md`.

Only after this tree exists may a catalog entry carry `validated_on` for this behaviour, and it must
cite the directory. Until then WS-1.2 is **unverified in both directions** — the third-party 23.6 GB/s
is unverified, and so is our §2 reading of it.

---

## 10 · Recon already done (2026-09-20, read-only)

Captured from this Mac over Tailscale; nothing was installed, started, stopped or changed.
Commands used: `ls /sys/class/infiniband/`, `rdma link show`, `ibv_devinfo -l`,
`cat /sys/class/infiniband/*/{board_id,fw_ver,node_guid,sys_image_guid,ports/1/{state,phys_state,rate,link_layer}}`,
`lspci -Dnn`, `ip -4 -br addr`, `ip -d link show`, `ethtool -i`, `ethtool <if> | grep -i speed`,
`cat /sys/class/net/*/{phys_port_name,phys_switch_id,carrier,speed,operstate,address,statistics/{rx,tx}_bytes}`.

`ethtool -m` (QSFP module EEPROM — the one read that would identify a physical cage by vendor serial
and settle §1 beyond inference) returned nothing as an unprivileged user. **Re-run it with privilege
when you are at the hardware:** matching `Vendor SN` across two netdevs is direct proof that they
share a cage, and would confirm or overturn §1 outright. Record it in `fabric.txt`.

---

## 11 · Provenance

The hypothesis under test, the `NCCL_IB_HCA` / `NCCL_CROSS_NIC=1` pairing, and the 23.6 GB/s figure come
from the third-party 8×DGX-Spark repo `glm-5.3-uncensored-8x-dgx-spark`
(`launchers/launch-nvfp4-tp8.sh:82-85`, `tools/run_nccl8.sh`, `tools/nccl_allreduce.py`), MIT licensed,
Copyright (c) 2026 im0xMagnus. Those numbers are facts and flag names, cited here as such; no prose
from that repo is reproduced. `scripts/hca_ab_bench.py` is an independent implementation whose
all-reduce timing approach is derived from `tools/nccl_allreduce.py` and carries the corresponding
provenance header. Full attribution belongs in `THIRD-PARTY-LICENSES` (WS-0.2).
