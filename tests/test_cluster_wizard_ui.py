"""เทส wizard ตั้งค่าเครือข่าย cluster (ConnectX-7 QSFP ของ DGX Spark) บนหน้าเว็บ — รัน JS จริงของ index.html

เทียบเท่า Cluster Assistant ของ NVIDIA Sync: เลือกเครื่อง → ตรวจสาย → แผน IP/netplan → apply (sudo) → ตรวจผล
backend (`/api/cluster/inspect|plan|apply|remove-net` ใน api.py + nodes/netplan.py) ทำคู่ขนานกัน จึงใช้ fetch
ปลอมที่ตอบ *รูปเดียวกับ backend จริง* (inspect_nodes / build_plan / apply_plan / remove_net) — เทสชุดนี้ยืนยัน
ฝั่งหน้าเว็บ: gating ของแต่ละขั้น · ตารางแผนวาดจาก payload จริง · รหัส sudo ส่งครั้งเดียวเฉพาะเครื่องที่เลือก
ไม่ลง localStorage · ขั้นตรวจผลแยกผ่าน/ล้มด้วยสี · ถอดเครือข่ายต้อง confirm ก่อน

ใช้ harness เดียวกับ test_console_shell (tests/console_shell_dom.js) · ไม่มี node = ข้าม
"""

from __future__ import annotations

from tests.test_console_shell import run_scenario  # noqa: F401 — harness + node lookup ร่วมกัน

# ฟลีต: Spark 3 เครื่องที่พร้อม · เครื่อง RTX (ไม่ใช่ Spark) · เครื่องที่ต่อไม่ติด
FLEET = """const fx = { nodes: [
  { name: "spark-01", site: "Lab" },
  { name: "spark-02", site: "Lab" },
  { name: "spark-03", site: "Lab" },
  { name: "rtx-box", site: "Lab", gpu: { name: "NVIDIA RTX 5090", vram_gb: 32, vram_used_gb: 1 } },
  { name: "down-01", site: "Lab", reachable: false, error: "ssh: connect timed out" } ],
  cluster: { machines: [
    { name: "spark-01", reachable: true, ready: false, has_gpu: true, fabric: { best_gbps: 200, tier: "rdma" } },
    { name: "spark-02", reachable: true, ready: false, has_gpu: true, fabric: { best_gbps: 200, tier: "rdma" } },
    { name: "spark-03", reachable: true, ready: false, has_gpu: true, fabric: { best_gbps: 200, tier: "rdma" } },
    { name: "rtx-box", reachable: true, ready: false, has_gpu: true, fabric: { best_gbps: 10, tier: "ethernet" } },
    { name: "down-01", reachable: false, ready: false, has_gpu: false, error: "ssh: connect timed out" } ], groups: [] } };
H.fx = fx;

// สายที่ "เสียบอยู่" ต่อเครื่อง/port — เทสสลับได้เพื่อจำลองสายหลุดแล้วกด Re-check
H.cables = { "spark-01": { 1: true, 2: false }, "spark-02": { 1: true, 2: false }, "spark-03": { 1: true, 2: true } };
H.speeds = { "spark-01": 200, "spark-02": 200, "spark-03": 200 };
// รูปเดียวกับ group_qsfp_ports ฝั่ง hub (profiler.py) — ที่ inspect_nodes และ inventory ใช้ร่วมกัน
const port = (n, p) => ({ port: p, ifaces: [`enp${p}s0f0np0`, `enp${p}s0f1np1`], carrier: !!H.cables[n][p],
  speed_gbps: H.cables[n][p] ? H.speeds[n] : null,
  configured: n === "spark-01" && p === 1 ? `enp${p}s0f1np1` : "", ip: n === "spark-01" && p === 1 ? "10.100.152.1" : "",
  prefix: n === "spark-01" && p === 1 ? 30 : null, rdma_devices: [`rocep${p}s0f0`, `rocep${p}s0f1`], netplan_managed: n === "spark-01" });
H.fabricOf = n => ({ best_gbps: 200, tier: "rdma", links: [], qsfp_ports: [1, 2].map(p => port(n, p)), netplan_files: [], nvidia_sync_netplan: false });
const inspect = (url, opts) => {
  const names = JSON.parse(opts.body).nodes;
  const nodes = {};
  for (const n of names) nodes[n] = { reachable: true, error: "", hostname: n, spark: true, ports: [1, 2].map(p => port(n, p)),
                                      sudo_needed: !(H.sudoFree && H.sudoFree.includes(n)),   // H.sudoFree = เครื่อง NOPASSWD
                                      netplan_files: [], nvidia_sync: false };
  const kind = names.length === 2 ? "direct-2" : names.length === 3 ? "ring-3" : "switch-4";
  return { nodes, topology: { topology: kind, links: [], reason: "", order: names },
           findings: [{ level: "warn", kind: "speed", text: "spark-02 port 1 negotiated 200G, fine" }], ok: true };
};
// รูปของ build_plan: links[].ends + nodes{}.links/netplan/cluster_ip + registry
const plan = (url, opts) => {
  const b = JSON.parse(opts.body);
  const base = (b.base_subnet || "10.100.152.0/24").split("/")[0].replace(/0$/, "");
  const [a, c] = b.nodes;
  const link = (me, peer, ip, peerIp) => ({ iface: "enp1s0f1np1", ip, prefix: 30, peer_node: peer, peer_ip: peerIp, link_id: "L1", qsfp_port: 1 });
  const yaml = ip => "network:\\n  version: 2\\n  renderer: networkd\\n  ethernets:\\n    enp1s0f1np1:\\n      addresses: [" + ip + "/30]";
  return { ok: true, topology: "direct-2", reason: "", order: [a, c], base_subnet: b.base_subnet, warnings: b.base_subnet === "10.200.0.0/24" ? ["10.200.0.0/24 overlaps the LAN of spark-02"] : [],
    links: [{ id: "L1", subnet: base + "0/30", ends: [
      { node: a, port: 1, iface: "enp1s0f1np1", ip: base + "1", prefix: 30, changed: true },
      { node: c, port: 1, iface: "enp1s0f1np1", ip: base + "2", prefix: 30, changed: true } ] }],
    nodes: { [a]: { links: [link(a, c, base + "1", base + "2")], netplan: yaml(base + "1"), cluster_ip: base + "1", cluster_iface: "enp1s0f1np1", changed: true },
             [c]: { links: [link(c, a, base + "2", base + "1")], netplan: yaml(base + "2"), cluster_ip: base + "2", cluster_iface: "enp1s0f1np1", changed: true } },
    registry: { [a]: { cluster_ip: base + "1", cluster_iface: "enp1s0f1np1", cluster_links: [] },
                [c]: { cluster_ip: base + "2", cluster_iface: "enp1s0f1np1", cluster_links: [] } } };
};
// step ตามที่ apply_plan / pair_workers รายงานจริง
const S = (node, step, ok = true, detail = "", level = "") => ({ node, step, ok, detail, level: level || (ok ? "pass" : "fail") });
H.stepsPartial = [S("spark-01", "sudo password accepted"), S("spark-02", "sudo password accepted"),
  S("spark-01", "stage netplan file", true, "/tmp/lmds-netplan.Ab12"), S("spark-01", "write /etc/netplan/90-lmds-cluster.yaml + netplan apply", true, "backup 90-lmds-cluster.yaml.20260905"),
  S("spark-01", "verify addresses + carrier", true, "enp1s0f1np1 10.100.152.1/30 LOWER_UP")];
H.pairing = [S("spark-01", "cluster key on spark-01 (~/.ssh/id_lmds_cluster)", true, "AAAAC3"),
  S("spark-02", "authorize spark-01 on spark-02 (~/.ssh/authorized_keys)"),
  S("spark-01", "ssh config on spark-01 for 10.100.152.2 (~/.ssh/config)"),
  S("spark-01", "spark-01 → user@10.100.152.2 without a password")];
H.pings = [{ node: "spark-01", iface: "enp1s0f1np1", peer_node: "spark-02", peer_ip: "10.100.152.2", link_id: "L1", ok: true },
           { node: "spark-02", iface: "enp1s0f1np1", peer_node: "spark-01", peer_ip: "10.100.152.1", link_id: "L1", ok: true }];
H.speed = [{ link_id: "L1", from: "spark-01", to: "spark-02", gbps: 98.4, skipped: "" }];
H.stepsAll = () => [...H.stepsPartial,
  S("spark-02", "stage netplan file", true, "/tmp/lmds-netplan.Cd34"), S("spark-02", "write /etc/netplan/90-lmds-cluster.yaml + netplan apply"),
  S("spark-02", "verify addresses + carrier", true, "enp1s0f1np1 10.100.152.2/30 LOWER_UP"),
  ...H.pings.map(p => S(p.node, `ping ${p.peer_node} ${p.peer_ip} via ${p.iface}`, p.ok, p.ok ? "" : "no reply — if the cabling is crossed (port 1 ↔ port 2) swap the cable")),
  ...H.pairing, ...H.speed.map(s => S(s.from, `iperf3 → ${s.to} 10.100.152.2`, true, `${s.gbps} Gbit/s`, s.gbps < 90 ? "warn" : "pass")),
  S("spark-01", "registry updated", true, "cluster_ip 10.100.152.1 on enp1s0f1np1"), S("spark-02", "registry updated", true, "cluster_ip 10.100.152.2 on enp1s0f1np1")];
H.result = () => ({ ok: H.pings.every(p => p.ok) && H.pairing.every(s => s.ok), applied: true, steps: H.stepsAll(), nodes: {}, pings: H.pings, pairing: H.pairing, speed: H.speed,
  registry: { "spark-01": { cluster_ip: "10.100.152.1", cluster_iface: "enp1s0f1np1" }, "spark-02": { cluster_ip: "10.100.152.2", cluster_iface: "enp1s0f1np1" } } });
H.jobPolls = 0;
H.doctor = { ok: true, findings: [{ level: "pass", kind: "gpu", text: "GPU present on both" },
                                  { level: "pass", kind: "link", text: "200G RDMA link on both" }] };
H.routes = [
  ["/api/cluster/inspect", inspect],
  ["/api/cluster/plan", plan],
  ["/api/cluster/apply", (url, opts) => { H.applyBody = JSON.parse(opts.body); return { id: "job-77", running: true, steps: [], result: null }; }],
  [/^\\/api\\/cluster\\/apply\\/job-77/, () => H.jobPolls++ === 0
     ? { id: "job-77", running: true, steps: H.stepsPartial, result: null }
     : { id: "job-77", running: false, steps: H.stepsAll(), result: H.result() }],
  ["/api/cluster/remove-net", (url, opts) => { H.removeBody = JSON.parse(opts.body); return { node: H.removeBody.node, ok: true, removed: true, absent: false,
     steps: [S(H.removeBody.node, "sudo password accepted"), S(H.removeBody.node, "move /etc/netplan/90-lmds-cluster.yaml to /root/netplan-disabled + netplan apply"),
             S(H.removeBody.node, "registry cleared")] }; }],
  [/^\\/api\\/cluster\\/doctor/, () => H.doctor],
  // inventory ของ harness ไม่มี fabric — เติม qsfp_ports ให้ (แผง Ports บนการ์ดอ่านจากตรงนี้)
  ...H.defaultRoutes(fx).map(([p, h]) => p instanceof RegExp && String(p).includes("inventory")
    ? [p, (...a) => { const r = h(...a); if (r && r.host && r.name.startsWith("spark")) r.host.fabric = H.fabricOf(r.name); return r; }] : [p, h]),
];
"""

# ตัวช่วยฝั่ง scenario: อ่านสถานะของ wizard ออกมาเป็น JSON
HELPERS = """
const text = el => (el ? el.textContent : "").replace(/\\s+/g, " ").trim();
const body = () => document.getElementById("cnw-body");
const nextBtn = () => document.querySelector('#cnw button[data-cnw="next"]');
const pick = async name => { const cb = document.querySelector(`#cnw input.cnw-pick[data-node="${name}"]`);
  cb.checked = true; cb.dispatchEvent(new Event("change", { bubbles: true })); await H.tick(); };
const open = async () => { document.getElementById("cluster-net-setup").click(); await H.tick(); };
const next = async () => { nextBtn().click(); await H.tick(12); };
const storedText = () => { const out = []; for (let i = 0; i < localStorage.length; i++) { const k = localStorage.key(i); out.push(k + "=" + localStorage.getItem(k)); } return out.join("\\n"); };
"""


def test_wizard_opens_and_greys_out_machines_that_cannot_join(tmp_path):
    """ปุ่ม "Set up cluster network" เปิด wizard: Spark ที่ต่อได้เลือกได้ · RTX/เครื่องล่มถูกปิดพร้อมเหตุผล ·
    ปุ่ม Next เปิดเมื่อเลือกครบ 2–4 เท่านั้น"""
    (out,) = run_scenario(tmp_path, FLEET, HELPERS + """
        await open();
        const rows = document.querySelectorAll("#cnw .cnw-table tbody tr").map(tr => ({
          name: text(tr.querySelector("b")), disabled: tr.querySelector("input").hasAttribute("disabled"),
          why: text(tr.querySelector("td:last-child")) }));
        const before = nextBtn().hasAttribute("disabled");
        await pick("spark-01");
        const one = nextBtn().hasAttribute("disabled");
        await pick("spark-02");
        console.log(JSON.stringify({ shown: H.visible(document.getElementById("cnw")), step: body().dataset.step, rows,
          before, one, two: nextBtn().hasAttribute("disabled"), alerts: H.alerts, errors: H.errors,
          title: text(document.querySelector("#cnw .cnw-box b")) }));""")
    assert out["shown"] and out["step"] == "devices" and out["errors"] == [] and out["alerts"] == []
    assert out["title"] == "Set up cluster network"
    by = {r["name"]: r for r in out["rows"]}
    assert not by["spark-01"]["disabled"] and not by["spark-02"]["disabled"] and by["spark-01"]["why"] == ""
    assert by["rtx-box"]["disabled"] and "not a DGX Spark (NVIDIA RTX 5090)" in by["rtx-box"]["why"]
    assert by["down-01"]["disabled"] and "unreachable" in by["down-01"]["why"]
    assert out["before"] and out["one"] and not out["two"], "Next ต้องเปิดเมื่อเลือก ≥2 เครื่อง"


def test_missing_cable_blocks_the_cabling_step_until_recheck(tmp_path):
    """สายหลุดที่ spark-02 → บอกเครื่อง/port ที่ขาด · Next ปิด · เสียบแล้วกด Re-check ถึงไปต่อได้
    inspect ต้องถูกยิงเฉพาะเครื่องที่เลือก · finding ของหมอเครือข่ายที่ไม่ใช่ pass ต้องโผล่"""
    (out,) = run_scenario(tmp_path, FLEET + 'H.cables["spark-02"] = { 1: false, 2: false };', HELPERS + """
        await open(); await pick("spark-01"); await pick("spark-02"); await next();
        const calls = () => H.calls.filter(c => c.url === "/api/cluster/inspect").map(c => JSON.parse(c.body).nodes);
        const blocked = { step: body().dataset.step, missing: text(body().querySelector(".cnw-missing")),
          next: nextBtn().hasAttribute("disabled"), topology: text(body().querySelector(".field b")),
          ports: body().querySelectorAll(".qsfp").map(q => q.className + ":" + text(q.querySelector(".tag"))),
          port1: text(body().querySelector('.qsfp[data-port="1"]')), calls: calls(),
          finding: text(body().querySelector(".warn.dim")) };
        H.cables["spark-02"] = { 1: true, 2: false };
        document.querySelector('#cnw button[data-cnw="recheck"]').click(); await H.tick(12);
        console.log(JSON.stringify({ blocked, after: { missing: body().querySelector(".cnw-missing") !== null,
          ok: text(body().querySelector(".ok-line")), next: nextBtn().hasAttribute("disabled"), calls: calls() },
          rules: text(body()).includes("3 machines → ring over both ports"), errors: H.errors }));""")
    b = out["blocked"]
    assert b["step"] == "cabling" and b["next"] and b["calls"] == [["spark-01", "spark-02"]]
    assert "spark-02: no cable detected on port 1 or port 2" in b["missing"]
    assert b["topology"] == "direct-2"
    assert b["ports"] == ["qsfp on:cable · 200G", "qsfp off:no cable", "qsfp off:no cable", "qsfp off:no cable"]
    assert "enp1s0f0np0 f0 no IP rocep1s0f0" in b["port1"] and "enp1s0f1np1 f1 10.100.152.1/30 rocep1s0f1" in b["port1"]
    assert b["port1"].endswith("netplan-managed") and b["finding"] == "! spark-02 port 1 negotiated 200G, fine"
    assert out["after"]["missing"] is False and not out["after"]["next"]
    assert "All required cables detected" in out["after"]["ok"]
    assert out["after"]["calls"] == [["spark-01", "spark-02"], ["spark-01", "spark-02"]], "Re-check ยิง inspect ซ้ำ"
    assert out["rules"] and out["errors"] == []


def test_three_machine_ring_needs_both_ports_on_every_machine(tmp_path):
    """กติกา NVIDIA: 3 เครื่อง = วงแหวน ต้องเสียบทั้ง port 1 และ 2 ทุกเครื่อง — spark-01/02 มีสายเดียวจึงถูกบล็อก"""
    (out,) = run_scenario(tmp_path, FLEET, HELPERS + """
        await open(); await pick("spark-01"); await pick("spark-02"); await pick("spark-03"); await next();
        console.log(JSON.stringify({ missing: text(body().querySelector(".cnw-missing")), next: nextBtn().hasAttribute("disabled"),
          note: text(body()).includes("needs 2 cabled ports per machine") }));""")
    assert out["next"] and out["note"]
    assert "spark-01: no cable detected on port 2 — a 3-machine ring needs both ports" in out["missing"]
    assert "spark-02: no cable detected on port 2" in out["missing"] and "spark-03" not in out["missing"]


def test_plan_table_renders_ips_from_the_payload_and_yaml_toggles(tmp_path):
    """ตารางลิงก์/ต่อเครื่องมาจาก payload ของ /api/cluster/plan (build_plan) ตรง ๆ · YAML ซ่อนไว้ กดแล้วโผล่ ·
    แก้ base subnet แล้ว Re-plan ต้องส่ง base_subnet ใหม่และวาด warning ที่ backend ตอบ ·
    plan ที่ ok:false (สายไม่ตรงผัง — HTTP 200) ต้องบอกเหตุผลและปิด Next"""
    (out,) = run_scenario(tmp_path, FLEET, HELPERS + """
        await open(); await pick("spark-01"); await pick("spark-02"); await next(); await next();
        const cells = tr => tr.querySelectorAll("td").map(text);
        const links = body().querySelectorAll(".cnw-links tbody tr").map(cells);
        const yaml = body().querySelector('pre.cnw-yaml[data-node="spark-02"]');
        const hiddenBefore = yaml.hidden;
        const tog = body().querySelector('button[data-cnw="yaml"][data-node="spark-02"]');
        tog.click(); await H.tick();
        const shown = { hidden: yaml.hidden, label: text(tog), text: yaml.textContent };
        tog.click(); await H.tick();
        const hiddenAgain = yaml.hidden;
        const nodes = body().querySelectorAll(".cnw-node").map(n => text(n.querySelector(".field")));
        const linkLines = body().querySelectorAll(".cnw-node .dim.mono").map(text);
        document.getElementById("cnw-subnet").value = "10.200.0.0/24";
        document.querySelector('#cnw button[data-cnw="replan"]').click(); await H.tick(12);
        const plans = H.calls.filter(c => c.url === "/api/cluster/plan").map(c => JSON.parse(c.body));
        const replanned = { warning: text(body().querySelector(".warn-line")), ip: cells(body().querySelector(".cnw-links tbody tr"))[4], next: nextBtn().hasAttribute("disabled") };
        H.routes.unshift(["/api/cluster/plan", () => ({ ok: false, topology: "unknown", reason: "spark-02 has no cabled port toward spark-01", order: ["spark-01", "spark-02"], warnings: [], nodes: {}, registry: {} })]);
        document.querySelector('#cnw button[data-cnw="replan"]').click(); await H.tick(12);
        console.log(JSON.stringify({ step: body().dataset.step, links, hiddenBefore, shown, hiddenAgain, nodes, linkLines, plans, replanned,
          notOk: { text: text(body().querySelector(".cnw-missing")), next: nextBtn().hasAttribute("disabled") }, errors: H.errors }));""")
    assert out["step"] == "plan" and out["errors"] == []
    assert out["links"] == [["L1", "10.100.152.0/30", "spark-01 port 1", "enp1s0f1np1", "10.100.152.1/30",
                             "spark-02 port 1", "enp1s0f1np1", "10.100.152.2/30"]]
    assert out["hiddenBefore"] is True and out["shown"]["hidden"] is False and out["hiddenAgain"] is True
    assert out["shown"]["label"] == "Hide netplan YAML" and "addresses: [10.100.152.2/30]" in out["shown"]["text"]
    assert out["nodes"][0].startswith("spark-01 head cluster IP 10.100.152.1 on enp1s0f1np1")
    assert out["nodes"][1].startswith("spark-02 cluster IP 10.100.152.2 on enp1s0f1np1")
    assert out["linkLines"] == ["enp1s0f1np1 (port 1) → 10.100.152.1/30 · peer spark-02 10.100.152.2",
                                "enp1s0f1np1 (port 1) → 10.100.152.2/30 · peer spark-01 10.100.152.1"]
    assert [p["base_subnet"] for p in out["plans"]][:2] == ["10.100.152.0/24", "10.200.0.0/24"]
    assert all(p["nodes"] == ["spark-01", "spark-02"] and p["topology"] == "direct-2" for p in out["plans"])
    assert out["replanned"] == {"warning": "10.200.0.0/24 overlaps the LAN of spark-02", "ip": "10.200.0.1/30", "next": False}
    assert out["notOk"]["next"] and "spark-02 has no cabled port toward spark-01" in out["notOk"]["text"]


def test_apply_sends_passwords_once_for_selected_nodes_only_and_never_stores_them(tmp_path):
    """รหัส sudo: ส่งไปกับ body ของ /api/cluster/apply เฉพาะเครื่องที่เลือก (rtx-box/down-01 ไม่ติดไป)
    ช่องเป็น type=password · ล้างหลังส่ง · ไม่มีอะไรลง localStorage · ติ๊กต่อเครื่องเดินตาม steps ของงาน
    (GET /api/cluster/apply/{id}) · งานจบ applied → ข้ามไปขั้น Verify เอง"""
    (out,) = run_scenario(tmp_path, FLEET, HELPERS + """
        await open(); await pick("spark-01"); await pick("spark-02"); await next(); await next(); await next();
        const step = body().dataset.step;
        const types = body().querySelectorAll("input.cnw-pw").map(i => i.type);
        // แยกรหัสต่อเครื่อง: ติ๊ก "one password" ออก → ช่องละเครื่อง
        const same = document.getElementById("cnw-samepw"); same.checked = false; same.dispatchEvent(new Event("change", { bubbles: true })); await H.tick();
        const perNode = body().querySelectorAll("input.cnw-pw").map(i => [i.dataset.node, i.type]);
        document.querySelector('#cnw button[data-cnw="apply"]').click(); await H.tick(4);
        const noPw = { warned: text(body().querySelector(".warn-line")), posted: H.calls.some(c => c.url === "/api/cluster/apply") };
        for (const i of body().querySelectorAll("input.cnw-pw")) i.value = "hunter2-" + i.dataset.node;
        document.querySelector('#cnw button[data-cnw="apply"]').click(); await H.tick(12);
        const ticks = n => body().querySelectorAll(`.cnw-prog-row[data-node="${n}"] .cnw-tick`).map(t => t.dataset.step + "=" + t.dataset.state);
        const running = { step: body().dataset.step, job: text(body().querySelector(".dim")), s1: ticks("spark-01"), s2: ticks("spark-02"),
          log: document.getElementById("cnw-log").textContent.split("\\n"),
          inputsLeft: document.querySelectorAll("#cnw input.cnw-pw").length,
          closeDisabled: document.querySelector('#cnw button[data-cnw="close"]').hasAttribute("disabled") };
        await H.sleep(1500); await H.tick(16);
        const verify = { step: body().dataset.step, done: text(body().querySelector(".ok-line")),
          pings: body().querySelectorAll(".cnw-ping tbody td.pass").map(text),
          ssh: body().querySelectorAll("#cnw-body > .pass").map(text),
          speed: text(body().querySelectorAll("#cnw-body > .pass").find(el => text(el).includes("iperf3"))),
          doctor: body().querySelectorAll(".cnw-node .pass").map(text) };
        console.log(JSON.stringify({ step, types, perNode, noPw, body: H.applyBody, running, verify, stored: storedText(),
          html: document.getElementById("cnw").innerHTML.includes("hunter2"), errors: H.errors, alerts: H.alerts,
          refreshed: H.calls.some(c => c.url === "/api/cluster?refresh=true") }));""")
    assert out["step"] == "apply" and out["types"] == ["password"]
    assert out["perNode"] == [["spark-01", "password"], ["spark-02", "password"]]
    assert "Enter the sudo password" in out["noPw"]["warned"] and out["noPw"]["posted"] is False
    assert out["body"]["passwords"] == {"spark-01": "hunter2-spark-01", "spark-02": "hunter2-spark-02"}
    assert out["body"]["plan"]["order"] == ["spark-01", "spark-02"] and out["body"]["plan"]["nodes"]["spark-01"]["cluster_ip"] == "10.100.152.1", \
        "ส่ง plan ทั้งก้อนกลับไปตามที่ apply_plan ต้องการ"
    assert "hunter2" not in out["stored"] and out["html"] is False, "รหัสผ่านต้องไม่ค้างทั้งใน DOM และ localStorage"
    r = out["running"]
    assert r["step"] == "apply" and "job-77" in r["job"] and r["inputsLeft"] == 0 and r["closeDisabled"]
    assert r["s1"] == ["sudo=ok", "netplan=ok", "addresses=ok", "firewall=wait", "ping=wait", "ssh=wait", "registry=wait"]
    assert r["s2"] == ["sudo=ok", "netplan=wait", "addresses=wait", "firewall=wait", "ping=wait", "ssh=wait", "registry=wait"]
    assert r["log"][0] == "✓ [spark-01] sudo password accepted" and r["log"][-1].startswith("✓ [spark-01] verify addresses + carrier — enp1s0f1np1 10.100.152.1/30")
    v = out["verify"]
    assert v["step"] == "verify"
    assert "Done — cluster IPs saved to the registry: spark-01 10.100.152.1 · spark-02 10.100.152.2" in v["done"]
    assert v["pings"] == ["✓ 10.100.152.2", "✓ 10.100.152.1"]
    assert "✓ spark-01 → user@10.100.152.2 without a password" in v["ssh"] and "✓ authorize spark-01 on spark-02 (~/.ssh/authorized_keys)" in v["ssh"]
    assert v["speed"] == "✓ iperf3 spark-01 → spark-02: 98.4 Gbit/s"
    assert v["doctor"][0].startswith("✓ spark-01 ⇄ spark-02 — ready for stacked")
    assert out["refreshed"], "จบแล้วต้องรีเฟรชการ์ดให้เห็น cluster IP ใหม่"
    assert out["errors"] == [] and out["alerts"] == []


def test_passwordless_sudo_machines_need_no_password_field(tmp_path):
    """เคสจริง msi-4/msi-5 (2026-09-05): msi-5 มี NOPASSWD · inspect ส่ง sudo_needed:false → ไม่มีช่องรหัสของเครื่องนั้น
    ไม่มีติ๊ก "one password" (เหลือเครื่องเดียวที่ต้องกรอก) · apply ส่งรหัสเฉพาะเครื่องที่ต้องใช้ · ทั้งคู่ NOPASSWD =
    ไม่มีช่องเลย กด Apply ได้ทันที · ขั้น Verify/ติ๊กเดินเหมือนเดิมเพราะ step ยังมีคำว่า sudo password"""
    (out,) = run_scenario(tmp_path, FLEET, HELPERS + """
        H.sudoFree = ["spark-02"];
        await open(); await pick("spark-01"); await pick("spark-02"); await next(); await next(); await next();
        const fields = body().querySelectorAll("input.cnw-pw").map(i => i.dataset.node);
        const same = !!document.getElementById("cnw-samepw");
        const note = text(body().querySelectorAll(".dim").find(el => text(el).includes("passwordless")));
        document.querySelector('#cnw button[data-cnw="apply"]').click(); await H.tick(4);
        const blocked = { warned: text(body().querySelector(".warn-line")), posted: H.calls.some(c => c.url === "/api/cluster/apply") };
        body().querySelector('input.cnw-pw[data-node="spark-01"]').value = "hunter2";
        document.querySelector('#cnw button[data-cnw="apply"]').click(); await H.tick(12);
        const posted = H.applyBody.passwords;
        await H.sleep(1500); await H.tick(16);
        const after = body().dataset.step;
        console.log(JSON.stringify({ fields, same, note, blocked, posted, after, errors: H.errors, alerts: H.alerts }));""")
    assert out["fields"] == ["spark-01"] and out["same"] is False
    assert out["note"] == "spark-02: passwordless sudo — no password needed"
    assert "spark-01" in out["blocked"]["warned"] and out["blocked"]["posted"] is False
    assert out["posted"] == {"spark-01": "hunter2"}, "เครื่อง NOPASSWD ต้องไม่มีรหัส (แม้ว่าง) ติดไปใน body"
    assert out["after"] == "verify" and out["errors"] == [] and out["alerts"] == []
    (both,) = run_scenario(tmp_path, FLEET, HELPERS + """
        H.sudoFree = ["spark-01", "spark-02"];
        await open(); await pick("spark-01"); await pick("spark-02"); await next(); await next(); await next();
        const fields = body().querySelectorAll("input.cnw-pw").length;
        document.querySelector('#cnw button[data-cnw="apply"]').click(); await H.tick(12);
        console.log(JSON.stringify({ fields, posted: H.applyBody && H.applyBody.passwords, step: body().dataset.step, errors: H.errors }));""")
    assert both["fields"] == 0 and both["posted"] == {} and both["step"] == "apply" and both["errors"] == []


def test_verify_step_renders_pass_and_fail_lines(tmp_path):
    """ผลตรวจที่ไม่ผ่านต้องเห็นเป็นบรรทัดล้ม/เตือนแยกสี: ping ทางเดียวล้ม · สาย 50G (สวิตช์ auto-neg) · iperf3 ต่ำ ·
    SSH ไม่ pair · doctor ตอบ not ready พร้อม fix · ปุ่ม "Run doctor again" ยิง doctor ซ้ำ —
    apply ที่ applied แต่ ok:false ยังต้องพาไป Verify (netplan อยู่แล้ว แค่ต้องดูว่าอะไรล้ม)"""
    prelude = FLEET + """
        H.speeds["spark-02"] = 50;
        H.pings[1].ok = false;
        H.pairing[3] = S("spark-01", "spark-01 → user@10.100.152.2 without a password", false, "Permission denied (publickey)");
        H.speed[0].gbps = 48.2;
        H.doctor = { ok: false, findings: [{ level: "pass", kind: "gpu", text: "GPU present on both" },
                                           { level: "warn", kind: "speed", text: "spark-02 link negotiated only 50G" },
                                           { level: "fail", kind: "ssh", text: "head cannot ssh into the worker", fix: "lmds cluster pair spark-01 spark-02" }] };
    """
    (out,) = run_scenario(tmp_path, prelude, HELPERS + """
        await open(); await pick("spark-01"); await pick("spark-02"); await next(); await next(); await next();
        body().querySelector("input.cnw-pw").value = "pw";
        document.querySelector('#cnw button[data-cnw="apply"]').click(); await H.tick(12);
        await H.sleep(1500); await H.tick(16);
        const cells = body().querySelectorAll(".cnw-ping tbody td").map(td => td.className + ":" + text(td));
        const speeds = body().querySelectorAll(".cnw-table:not(.cnw-ping) tbody td").map(td => td.className + ":" + text(td));
        const lines = body().querySelectorAll("#cnw-body > .pass, #cnw-body > .fail, #cnw-body > .warn").map(el => el.className + ":" + text(el));
        const doctor = body().querySelectorAll(".cnw-node > div").map(el => el.className + ":" + text(el));
        const before = H.calls.filter(c => c.url.startsWith("/api/cluster/doctor")).length;
        document.querySelector('#cnw button[data-cnw="doctor"]').click(); await H.tick(12);
        console.log(JSON.stringify({ step: body().dataset.step, cells, speeds, lines, doctor, before,
          after: H.calls.filter(c => c.url.startsWith("/api/cluster/doctor")).length, errors: H.errors }));""")
    assert out["step"] == "verify" and out["errors"] == []
    assert out["cells"] == [":spark-01", "dim:—", "pass:✓ 10.100.152.2", ":spark-02", "fail:✕ 10.100.152.1", "dim:—"]
    assert out["speeds"][0] == ":spark-01" and out["speeds"][1].startswith("pass:200G")
    assert out["speeds"][4].startswith("warn:50G (expected 200G — check the switch port)")
    assert "warn:! iperf3 spark-01 → spark-02: 48.2 Gbit/s — below the ~100 Gbit/s PCIe ceiling; check the switch port speed" in out["lines"]
    assert "fail:✕ spark-01 → user@10.100.152.2 without a password — Permission denied (publickey)" in out["lines"]
    assert "pass:✓ cluster key on spark-01 (~/.ssh/id_lmds_cluster)" in out["lines"]
    assert out["doctor"][0].startswith("fail:✕ spark-01 ⇄ spark-02 — not ready")
    assert any(d.startswith("warn dim:! spark-02 link negotiated only 50G") for d in out["doctor"])
    assert any(d.startswith("fail dim:✕ head cannot ssh into the worker lmds cluster pair") for d in out["doctor"])
    assert out["after"] == out["before"] + 1


def test_apply_failure_rolls_back_and_stays_on_the_apply_step(tmp_path):
    """รหัส sudo ผิด / netplan apply ล้ม → งานจบ applied:false — อยู่ที่ขั้น Apply พร้อมบรรทัดล้มและ rollback
    ปุ่ม Verify ปิด · กลับไปแก้แผนได้"""
    prelude = FLEET + """
        H.routes.unshift([/^\\/api\\/cluster\\/apply\\/job-77/, () => ({ id: "job-77", running: false,
          steps: [S("spark-01", "sudo password accepted"), S("spark-02", "sudo password accepted"),
                  S("spark-01", "stage netplan file", true, "/tmp/lmds-netplan.Ab12"),
                  S("spark-01", "write /etc/netplan/90-lmds-cluster.yaml + netplan apply", false, "netplan apply failed: enp1s0f1np1: cannot set address"),
                  S("spark-01", "rollback to the previous netplan", true, "")],
          result: { ok: false, applied: false, steps: [], nodes: { "spark-01": { ok: false, rolled_back: true } }, pings: [], pairing: [], speed: [], registry: {} } })]);
    """
    (out,) = run_scenario(tmp_path, prelude, HELPERS + """
        await open(); await pick("spark-01"); await pick("spark-02"); await next(); await next(); await next();
        body().querySelector("input.cnw-pw").value = "pw";
        document.querySelector('#cnw button[data-cnw="apply"]').click(); await H.tick(16);
        console.log(JSON.stringify({ step: body().dataset.step, status: text(body().querySelector(".dim")),
          ticks: body().querySelectorAll('.cnw-prog-row[data-node="spark-01"] .cnw-tick').map(t => t.dataset.step + "=" + t.dataset.state),
          log: document.getElementById("cnw-log").textContent.split("\\n"), warn: text(body().querySelector(".warn-line")),
          next: nextBtn().hasAttribute("disabled"), back: !!document.querySelector('#cnw button[data-cnw="back"]'), errors: H.errors }));""")
    assert out["step"] == "apply" and out["status"].endswith("failed — rolled back")
    assert out["ticks"][:3] == ["sudo=ok", "netplan=bad", "addresses=wait"]
    assert out["log"][3] == "✕ [spark-01] write /etc/netplan/90-lmds-cluster.yaml + netplan apply — netplan apply failed: enp1s0f1np1: cannot set address"
    assert out["log"][4] == "✓ [spark-01] rollback to the previous netplan"
    assert "rolled back" in out["warn"] and out["next"] and out["back"] and out["errors"] == []


def test_inspect_panel_and_remove_net_confirm(tmp_path):
    """ปุ่ม Ports บนแถบ cluster ของการ์ด → แผงดู QSFP ของเครื่องนั้นจาก inventory · Re-check ดึง inventory สด ·
    "Remove cluster network" ต้อง confirm: ตอบไม่ = ไม่ยิงอะไร · ตอบใช่ = ขอรหัส sudo ในฟอร์ม (type=password)
    แล้ว POST /api/cluster/remove-net ครั้งเดียว และโชว์ step ที่ backend ทำ"""
    (out,) = run_scenario(tmp_path, FLEET, HELPERS + """
        await H.go("#/nodes");
        const strip = nodeRows.get("spark-01").block.querySelector(".nclus");
        strip.querySelector('button[data-cact="inspect-net"]').click(); await H.tick(4);
        const inventoryCalls = () => H.calls.filter(c => c.url.startsWith("/api/nodes/spark-01/inventory?refresh=true")).length;
        const panel = { shown: H.visible(document.getElementById("cnw")), title: text(document.querySelector("#cnw .cnw-box b")),
          ports: body().querySelectorAll(".qsfp").map(q => text(q)),
          inspectCalls: H.calls.filter(c => c.url === "/api/cluster/inspect").length, live: inventoryCalls() };
        document.querySelector('#cnw button[data-cnw="inspect-recheck"]').click(); await H.tick(12);
        const rechecked = { live: inventoryCalls(), ports: body().querySelectorAll(".qsfp").length };
        H.confirmAnswer = false;
        document.querySelector('#cnw button[data-cnw="remove-net"]').click(); await H.tick();
        const declined = { confirms: H.confirms.length, form: document.querySelector("#cnw input.cnw-rm-pw") !== null,
          posted: H.calls.some(c => c.url === "/api/cluster/remove-net") };
        H.confirmAnswer = true;
        document.querySelector('#cnw button[data-cnw="remove-net"]').click(); await H.tick();
        const pw = document.querySelector("#cnw input.cnw-rm-pw");
        const form = { type: pw && pw.type, node: pw && pw.dataset.node };
        pw.value = "s3cret";
        document.querySelector('#cnw button[data-cnw="remove-go"]').click(); await H.tick(12);
        console.log(JSON.stringify({ panel, rechecked, declined, confirmText: H.confirms[1], form, body: H.removeBody,
          result: text(body().querySelector(".cnw-rm .pass")), steps: body().querySelectorAll(".cnw-rm + .dim, .cnw-rm ~ .dim").map(text),
          formGone: document.querySelector("#cnw input.cnw-rm-pw") === null,
          stored: storedText(), html: document.getElementById("cnw").innerHTML.includes("s3cret"), errors: H.errors, alerts: H.alerts }));""")
    p = out["panel"]
    assert p["shown"] and p["title"] == "Cluster network ports" and p["inspectCalls"] == 0 and p["live"] == 1
    assert len(p["ports"]) == 2 and "QSFP port 1 cable · 200G" in p["ports"][0] and "10.100.152.1/30" in p["ports"][0]
    assert "QSFP port 2 no cable" in p["ports"][1] and "netplan-managed" in p["ports"][0]
    assert out["rechecked"] == {"live": 2, "ports": 2}, "Re-check ต้องดึง inventory สด (refresh=true) แล้ววาดใหม่"
    assert out["declined"] == {"confirms": 1, "form": False, "posted": False}
    assert "Remove the cluster network from spark-01?" in out["confirmText"]
    assert out["form"] == {"type": "password", "node": "spark-01"}
    assert out["body"] == {"node": "spark-01", "password": "s3cret"}
    assert out["result"].startswith("✓ cluster network removed from spark-01") and out["formGone"]
    assert any("move /etc/netplan/90-lmds-cluster.yaml to /root/netplan-disabled" in s for s in out["steps"])
    assert "s3cret" not in out["stored"] and out["html"] is False
    assert out["errors"] == [] and out["alerts"] == []


def test_group_header_opens_the_wizard_with_its_members_preselected(tmp_path):
    """หัวกลุ่ม cluster มีปุ่มเดียวกัน — เปิดมาแล้วสมาชิกของกลุ่มถูกเลือกไว้ให้ ไม่ต้องติ๊กเอง"""
    prelude = FLEET + """
        fx.cluster.groups = [{ members: [{ name: "spark-01" }, { name: "spark-02" }], ready: false, gpu: "NVIDIA GB10", gpus_per_node: 1,
          world_size: 2, link_gbps: 200, rdma: true, site: "Lab", blockers: [], warnings: [], excluded: [] }];
    """
    (out,) = run_scenario(tmp_path, prelude, HELPERS + """
        await H.go("#/nodes");
        const btn = document.querySelector('#nodes .gbar button[data-cact="net-wizard"]');
        btn.click(); await H.tick();
        console.log(JSON.stringify({ nodes: btn.dataset.nodes, step: body().dataset.step,
          picked: document.querySelectorAll("#cnw input.cnw-pick").filter(i => i.hasAttribute("checked")).map(i => i.dataset.node),
          next: nextBtn().hasAttribute("disabled"), errors: H.errors }));""")
    assert out["nodes"] == "spark-01,spark-02" and out["step"] == "devices"
    assert out["picked"] == ["spark-01", "spark-02"] and not out["next"] and out["errors"] == []


def test_documented_contract_shapes_are_still_accepted(tmp_path):
    """สัญญาที่เขียนไว้ก่อน backend ลงมือ (inspect: fabric.ports[].interfaces[] + topology.kind ·
    plan: links[].a/b + per_node{iface_ips,netplan_yaml}) ต้องยังอ่านได้ — กัน UI พังถ้า payload สลับรุ่น"""
    prelude = FLEET + """
        H.routes.unshift(["/api/cluster/inspect", (url, opts) => { const names = JSON.parse(opts.body).nodes; const nodes = {};
          for (const n of names) nodes[n] = { reachable: true, sudo_needed: true, fabric: { ports: [1, 2].map(p => ({ qsfp_port: p, interfaces: [
            { iface: `enp${p}s0f1np1`, function: "f1", carrier: p === 1, speed_gbps: p === 1 ? 200 : 0, ip: "", prefix: null, rdma_device: `rocep${p}s0f1`, netplan_managed: false }] })), links: [] } };
          return { nodes, topology: { kind: "direct-2", reason: "", cabled: [] } }; }]);
        H.routes.unshift(["/api/cluster/plan", () => ({ topology: { kind: "direct-2" },
          links: [{ link_id: "L1", a: { node: "spark-01", iface: "enp1s0f1np1", ip: "10.100.152.1", prefix: 30 }, b: { node: "spark-02", iface: "enp1s0f1np1", ip: "10.100.152.2", prefix: 30 } }],
          per_node: { "spark-01": { iface_ips: [{ iface: "enp1s0f1np1", ip: "10.100.152.1", prefix: 30, peer_node: "spark-02", peer_ip: "10.100.152.2" }], netplan_yaml: "network: {}", cluster_ip: "10.100.152.1", changes: ["write netplan"] },
                      "spark-02": { iface_ips: [{ iface: "enp1s0f1np1", ip: "10.100.152.2", prefix: 30, peer_node: "spark-01", peer_ip: "10.100.152.1" }], netplan_yaml: "network: {}", cluster_ip: "10.100.152.2", changes: [] } },
          warnings: [] })]);
    """
    (out,) = run_scenario(tmp_path, prelude, HELPERS + """
        await open(); await pick("spark-01"); await pick("spark-02"); await next();
        const cabling = { ok: text(body().querySelector(".ok-line")), topology: text(body().querySelector(".field b")), next: nextBtn().hasAttribute("disabled") };
        await next();
        console.log(JSON.stringify({ cabling, step: body().dataset.step,
          links: body().querySelectorAll(".cnw-links tbody tr").map(tr => tr.querySelectorAll("td").map(text)),
          next: nextBtn().hasAttribute("disabled"), errors: H.errors }));""")
    assert out["cabling"]["topology"] == "direct-2" and "All required cables" in out["cabling"]["ok"] and not out["cabling"]["next"]
    assert out["step"] == "plan" and not out["next"] and out["errors"] == []
    assert out["links"] == [["L1", "—", "spark-01", "enp1s0f1np1", "10.100.152.1/30", "spark-02", "enp1s0f1np1", "10.100.152.2/30"]]
