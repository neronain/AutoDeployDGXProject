// Scenario for tests/test_audit_console_dom.py::test_no_payload_field_can_create_markup
//
// ทุก "ใบ" ของ payload ตัวอย่าง (node · model · bench · scan · job · error · cluster · fit · deploy plan …) ถูกแทนด้วย
// markup ทีละใบ แล้ววาดด้วยทางเดียวกับที่หน้าเว็บใช้จริง · ผ่าน = ไม่มี element/attribute ไหนเกิดจากค่านั้น
// เดิน payload เอง (ไม่ได้ไล่ชื่อฟิลด์) — ฟิลด์ที่เพิ่มเข้า payload ตัวอย่างทีหลังจึงถูกตรวจโดยปริยาย
//
// ตัว marker ออกแบบให้หลุดได้ทุกบริบทที่ template ใช้: ข้อความ (<img>) · attribute ใน "…" (") · ใน '…' (') ·
// attribute ไม่มี quote (ช่องว่าง + data-pwn=) · และถ้าไปอยู่ใน on*="…" ก็นับว่าหลุด (esc() กัน ' ไม่ได้ในบริบท JS)
H.fastTimers(200);
const MARK = path => `x"'><img src=x data-pwn="${path}">`;
H.MARK = MARK;

const fullModel = {
  slug: "full", model_id: "Org/Full-GGUF", engine: "llamacpp", mode: "native", port: 8080, endpoint: "http://10.0.0.1:8080/v1",
  context: 65536, context_per_request: 16384, context_configured: 32768, native_context: 131072, slots: 4, max_num_seqs: 4,
  running: true, healthy: true, downloaded: true, controller_exists: true, external: false, registered: true,
  self_managed_weights: false, autostart: "enabled", topology: "single", features: "tools, image", projector: true,
  speculative: true, feature_note: "mmproj note", served_name: "full-served", default_served_name: "full-default",
  commands: ["status", "bench", "test-text", "props", "wait-health", "verify-files", "check-runtime"],
  moe: { experts: 128, experts_active: 8 },
  pending_restart: { pending: true, changes: [{ field: "context", saved: "131072", running: "65536" }] },
  runtime_arch: { supported: false, runtime: "llama.cpp b6000", arch: "qwen4", fix: "update runtime", mode: "native" },
  runtime: { llamacpp_dir: "/opt/llama.cpp" },
  controller: { state: "stale", generated_by: "0.5.0", reason: "older template" },
  job: null,
};
const stackedHead = { slug: "pair", model_id: "Org/Pair", engine: "vllm", port: 8000, context: 262144, running: true, healthy: false,
  downloaded: true, controller_exists: true, topology: "stacked", stacked_role: "head", stacked_peers: ["n2"],
  cluster: { nnodes: 2, worker_ips: ["10.100.152.2"] }, commands: ["status", "sync-worker", "verify-worker"], autostart: "n/a" };
const stackedShadow = { slug: "pair2", model_id: "Org/Pair", engine: "vllm", port: 8001, context: 262144, running: true, healthy: true,
  downloaded: true, topology: "stacked", stacked_role: "worker", stacked_head: "n9", commands: [], controller_exists: false,
  registered: false, external: false, autostart: "n/a", job: null };
const rivalModel = { slug: "rival", model_id: "Org/Rival", engine: "llamacpp", port: 8080, context: 8192, running: false, healthy: false,
  downloaded: false, controller_exists: true, commands: ["status"], autostart: "disabled", self_managed_weights: true };

const host = {
  hostname: "n1-host", lmds_version: "0.10.0", lmds_commit: "abc1234", ip: "10.0.0.1", site: "TKC",
  ips: [{ iface: "eth0", ip: "10.0.0.1", prefix: 24, primary: true, link_local: false }, { iface: "enp1s0f0", ip: "169.254.1.1", prefix: 16, primary: false, link_local: true }],
  gpus: [{ name: "NVIDIA RTX 3060", vram_gb: 12, vram_used_gb: 4.5, compute: "8.6", tested: true, pcie_gen: 4, pcie_width: 16,
           utilization_pct: 37, power_w: 120.5, power_limit_w: 170, temperature_c: 61, fan_pct: 40,
           clock_graphics_mhz: 1800, clock_graphics_max_mhz: 2100, clock_memory_mhz: 7000, clock_sm_mhz: 1800 },
         { name: "NVIDIA RTX 3060", vram_gb: 12, vram_used_gb: null, compute: "8.6", tested: false }],
  cpu: { cores: 20, load1: 1.25, percent: 12.5 }, memory_model: "discrete",
  ram_used_gb: 30.5, ram_total_gb: 128, disk_free_gb: 800, disk_total_gb: 2000,
  fabric: { best_gbps: 200, tier: "rdma", summary: "200G RDMA" },
  docker: true, toolkit: true, profile: "spark", arch: "arm64",
  docker_access: { user: "u", usable: false, installed: true, reason: "not in the docker group", fix: "sudo usermod -aG docker u" },
  cache: { exists: true, owner_ok: false, writable: false },
  runtimes: { llamacpp: [{ present: true, dir: "/opt/llama.cpp", commit: "deadbee", build: "b6000", date: "2026-09-01", lock_state: "stale" }] },
  role: { control_plane: false, engines: ["llamacpp", "vllm"], evidence: "GPU 2" },
};
H.nodeData = { host, models: [fullModel, stackedHead, stackedShadow, rivalModel], summary: { running: 3, total: 4 } };
H.localData = { host: { ...host, hostname: "hub-host", role: { control_plane: true, engines: [], evidence: "no GPU" } },
  models: [{ ...fullModel, slug: "local-full", job: null }, { ...stackedHead, slug: "local-pair" }, { ...rivalModel, slug: "local-rival", controller_exists: false }] };
H.benchFleet = { runs: [{ slug: "full", model_id: "Org/Full-GGUF", machine_name: "n1", hostname: "n1-host", engine: "llamacpp",
    stamped_at: "2026-10-01T10:00:00", speed_from: "2026-10-01T09:00:00", probes_from: "2026-10-01T09:30:00",
    environment: { engine_build: "b6000", context: 65536, quant: "Q4_K_M" },
    speed: { decode_tps_avg: 42.5, decode_tps_long: 30.1, longest_context: 8192, ttft_s_short: 0.25, failed: 1 },
    capability: { score: 85, passed: ["thai", "json"], failed: ["tools"], skipped: ["vision"] } }],
  unreachable: [{ node: "n2", error: "ssh timeout" }] };
H.benchRun = { run: { slug: "full", model_id: "Org/Full-GGUF", served_name: "full-served", engine: "llamacpp", stamped_at: "2026-10-01T10:00:00",
    cached_note: "(cached)", environment: { engine_build: "b6000", context: 65536, quant: "Q4_K_M" },
    machine: { hostname: "n1-host", cpu: "Cortex-X925", ram_total_gb: 128, gpus: [{ name: "NVIDIA GB10", vram_gb: 128 }] },
    workloads: [{ label: "512", target_input: 512, completion_tokens: 128, prompt_tokens: 520, decode_tps: 41.2, prefill_tps: 900.4, ttft_s: 0.31 },
                { label: "8k", target_input: 8192, completion_tokens: 128, prompt_tokens: 8200, decode_tps: 30.5, prefill_tps: 700, ttft_s: 2.5, error: "" }],
    probes: [{ key: "thai", passed: true, skipped: false, detail: "" }, { key: "tools", passed: false, skipped: false, detail: "no tool_calls" },
             { key: "vision", passed: false, skipped: true, detail: "" }] } };
H.scan = { n1: [{ kind: "gguf", name: "Org/Full-GGUF", layout: "root", size_gb: 18.5, shards: 3 }], "this machine": [] };
H.job = { id: "jx", command: "repair", running: false, elapsed: 75, output: "sha256 mismatch\n", exit_code: 1, slug: "full", node: "n1", steps: ["download", "verify-files"] };
H.cluster = { live: false,
  machines: [
    { name: "hub-host", self: true, reachable: true, stack: true, candidate: true, reason: "", hostname: "hub-host", ready: true, has_gpu: true,
      fabric: { best_gbps: 200, tier: "rdma", summary: "200G" }, cluster_ip: "10.100.152.9", cluster_name: "", suggested_ip: "10.100.152.9",
      ip: { state: "ok", iface: "enp1s0f0", speed_gbps: 200 } },
    { name: "n1", self: false, reachable: true, stack: true, candidate: true, hostname: "other-name", ready: false, has_gpu: true,
      fabric: { best_gbps: 50, tier: "eth" }, cluster_ip: "10.100.152.1", cluster_name: "rack-a", suggested_ip: "10.100.152.1",
      ip: { state: "slow", iface: "enp1s0f0", speed_gbps: 10, warning: { kind: "under-negotiated", speed_gbps: 50, expected_gbps: 200 } } },
    { name: "n2", self: false, reachable: false, ready: false, hostname: "", has_gpu: false, error: "ssh: no route", fabric: null, stack: false,
      cluster_ip: "", cluster_name: "", suggested_ip: "", ip: { state: "unset", iface: "", speed_gbps: null } }],
  groups: [{ members: [{ name: "n1" }, { name: "n2" }], ready: true, site: "TKC", cluster_name: "rack-a", gpu: "NVIDIA GB10", gpus_per_node: 1,
    world_size: 2, usable_world_size: 1, link_gbps: 200, rdma: true, fabric_network: "10.100.152.0/24",
    parallelism: { kind: "pipeline-parallel", largest_tp: 2 },
    blockers: [{ kind: "missing-ip", names: ["n2"] }],
    warnings: [{ kind: "under-negotiated", speed_gbps: 50, expected_gbps: 200, names: ["n1"] },
               { kind: "needs-switch", node_count: 4, max_direct: 3, names: ["n1"], shopping: ["a switch"] }, { kind: "other", names: ["n1"] }],
    excluded: [{ name: "n3", reason: "same-machine", same_as: "n1" }, { name: "n4", reason: "no-subnet" }] }] };
H.fitPlan = { verdict: "no-fit", reason: "", total_gb: 128, usable_gb: 120, os_reserve_gb: 8, weights_gb: 80.2, weights_source: "measured",
  overhead_gb: 4, kv_per_request_gb: 3.25, kv_source: "profile", kv_dtype: "bf16", kv_gb: 13, kv_pin_gb: 15.6, kv_pin_bytes: 16750372454,
  pin_supported: true, slots: 4, context: 65536, per_slot_context: 16384, ram_needed_gb: 99.8, ram_after_gb: -3.2, others_running_gb: 20,
  others: [{ slug: "other", gb: 20 }], running: true, own_gb_now: 90, stop_suggestion: "other", suggested_slots_max: 2,
  suggested_context_max: 32768, gpu_util_equivalent: 0.71, current_pin_bytes: 1, notes: ["a note"],
  stack: [{ kind: "os", label: "OS", gb: 8 }, { kind: "other", label: "other", gb: 20 }, { kind: "weights", label: "weights", gb: 80 }] };
H.plan = { id: "s1", notes: ["hub note"], plan: { model_id: "Org/Full", engine: "vllm", image: "vllm/vllm-openai:v1", engine_reason: "safetensors",
    revision: "abc", generator: "rules", context: 32768, selected_gguf: "model.gguf", task: "rerank",
    recipe: { label: "Proven", validated_on: "dgx-spark", controller: "x-single.sh" },
    fit: { target: "dgx-spark-single", verdict: "fits", budget_gb: 100, capacity_gb: 128, weights_gb: 20, reserved_gb: 30, reserved_source: "n1",
           running_now: ["other@n1"], max_safe_context: 131072, now_max_safe_context: 16384, kv_at_context_gb: 90, notes: ["fit note"] },
    capabilities: { tool_calling: { status: "yes", evidence: "chat template", caveat: "needs parser" }, vision: { status: "no", evidence: "none" } },
    warnings: ["plan warning"], flags_needing_approval: ["--trust-remote-code"] } };
H.advice = { available: true, asked: 32768, kv_dtype: "bf16", kv_bytes_per_token: 100000,
  ladder: [{ context: 32768, fits: true, kv_gb: 3.1, concurrency: 4.5 }, { context: 131072, fits: false, kv_gb: 12, concurrency: 0 }],
  advice: [{ kind: "single-user", level: "warn", facts: { concurrency: 1.2 } }, { kind: "room-to-grow", level: "ok", facts: { suggest: 65536, concurrency: 2 } }] };
H.built = { slug: "full", gates: [1, 2, 3], context: 32768, zip: "/bundles/full.zip", api_key: "sk-test", bundle: "/bundles/full" };
H.summary = { machines: 3, online: 2, pending: 1, gpus: 4, vram_gb: 280, models_running: 2, models_healthy: 1, models_total: 5 };
H.consistency = { summary: { consistent: 1, total: 2, controllers_stale: 1, runtime_stale: 1, line: "1 of 2 match" },
  hub: { dirty: ["a.py"], verdict: { level: "warn", controllers: { state: "stale", detail: "1 stale", items: ["local-full"] }, runtimes: { state: "ok", detail: "" } } },
  nodes: [{ name: "n1", level: "bad", consistent: false, source: "registry", code: { state: "ok", detail: "same commit" },
            controllers: { state: "stale", detail: "1 older", items: ["full"] }, runtimes: { state: "unknown", detail: "not downloaded", items: [] } }] };
H.recipes = { recipes: [{ match: "Org/Full", label: "Full", engine: "vllm", topology: "single", image: "img", validated_on: "dgx-spark",
    controller: "x-single.sh", source: "https://example.org/x", tools: "hermes", reasoning: "qwen3", serving: { context: 1 } }],
  source: { repo: "org/controllers", commit: "abc", count: 1, synced_at: "2026-10-01" }, default_repo: "org/controllers" };

const fx = { nodes: [{ name: "n1", site: "TKC", host: "10.0.0.1" }], localModels: H.localData.models, host: H.localData.host };
H.fx = fx;
H.served = {};          // url → payload ที่ scenario ตั้งให้ตอบรอบนี้ (ไม่ตั้ง = ค่าปกติของ fixture)
const dyn = (key, fallback) => () => (key in H.served ? H.served[key] : fallback());
H.routes = [
  ["/api/bench/fleet", dyn("bench", () => H.benchFleet)],
  [/^\/api\/nodes\/n1\/bench\//, dyn("benchRun", () => H.benchRun)],
  ["/api/scan", dyn("scan", () => H.scan)],
  [/^\/api\/jobs\/[^/]+$/, () => { const hit = H.served.job; delete H.served.job; return hit || { status: 404, body: { detail: "gone" } }; }],
  ["/api/cluster", dyn("cluster", () => ({ machines: [], groups: [] }))],
  [/^\/api\/deploy\/s1\/context/, dyn("advice", () => ({ available: false }))],
  ["/api/recipes", dyn("recipes", () => H.recipes)],
  ["/api/targets", () => ({ targets: [] })],
  [/\/clone\/targets$/, () => ({ targets: [] })], [/\/fit$/, () => ({ status: 409, body: { detail: "n/a" } })],
  [/^\/api\/models\/[^/]+\/(doctor|logs|key)/, dyn("localPanel", () => ({ findings: [], text: "", has_key: false }))],
  ["/api/deploy/analyze", dyn("analyze", () => H.plan)],
  ["/api/nodes", (url, opts) => (opts.method === "POST" && "addNode" in H.served) ? H.served.addNode
     : { nodes: fx.nodes.map(n => ({ name: n.name, site: n.site, user: "u", host: n.host, port: 22, note: "", local_ip: n.host })) }],
  ...H.defaultRoutes(fx),
];
// ---- boot ----
await H.tick(20);
const MARK = H.MARK;
const clone = v => JSON.parse(JSON.stringify(v));
const leaves = (v, path = []) => (v && typeof v === "object")
  ? Object.keys(v).flatMap(k => leaves(v[k], [...path, k])) : [path];
const withLeaf = (base, path, value) => { const copy = clone(base); let at = copy; for (const k of path.slice(0, -1)) at = at[k]; at[path[path.length - 1]] = value; return copy; };
const scratch = document.createElement("div"); scratch.id = "xss-scratch"; document.body.appendChild(scratch);
// สิ่งที่เกิดจาก marker: element/attribute ชื่อ data-pwn · หรือ marker ไปอยู่ใน inline handler (on*)
const offenders = () => {
  const out = [];
  for (const el of document.querySelectorAll("*")) for (const a of el.attributes) {
    if (a.name === "data-pwn") out.push(`<${el.tagName.toLowerCase()}> from ${a.value}`);
    else if (a.name.startsWith("on") && a.value.includes("data-pwn")) out.push(`${a.name}= handler`);
  }
  return out;
};
const report = {};
async function walk(surface, base, render) {
  const paths = leaves(base), found = [], threw = [];
  for (const path of paths) {
    const name = path.join(".");
    try { await render(withLeaf(base, path, MARK(name))); } catch (err) { threw.push(`${name}: ${String(err && err.message || err).slice(0, 90)}`); }
    const hit = offenders();
    if (hit.length) found.push(`${name} → ${[...new Set(hit)].slice(0, 2).join(" | ")}`);
    scratch.innerHTML = "";
  }
  try { await render(clone(base)); } catch (err) { threw.push(`(clean payload): ${String(err && err.message || err).slice(0, 90)}`); }
  report[surface] = { leaves: paths.length, found, threw };
}

// ── node ที่ถูกยึด: frame ของ SSE → การ์ด (ทั้งปิดและกางเมนูของทุกโมเดล) · ตาราง Fleet models · ภาพรวม · rail ──
location.hash = "#/models"; await H.tick(6);
await walk("node card + fleet models + overview", H.nodeData, async data => {
  for (const key of [...openModelMenus]) openModelMenus.delete(key);
  applySnapshot({ nodes: { n1: { age_seconds: 2, data } } });
  for (const b of [...nodeRows.get("n1").body.querySelectorAll('button[data-nact="menu"]')]) openModelMenus.add("n1/" + b.dataset.slug);
  paintNodeBody(nodeRows.get("n1"), "n1", lastNodeData.get("n1"));
  renderAllModels(); route = { kind: "overview" }; renderOverview(true); route = parseRoute();
  if (!document.querySelector(".pal-back")) palOpen();
  palRender("");
});
palClose();
await walk("node unreachable", { error: "ssh: connect to host 10.0.0.1 port 22: No route to host", age_seconds: 12 }, async entry => {
  for (const key of [...openModelMenus]) openModelMenus.delete(key);
  applySnapshot({ nodes: { n1: entry } });
  route = { kind: "overview" }; renderOverview(true); route = parseRoute();
});
applySnapshot({ nodes: { n1: { age_seconds: 2, data: clone(H.nodeData) } } });

// ── โมเดลในเครื่องนี้ + host ของ hub: การ์ด · แผง Manage / Tests / Key / Score ──
await walk("hub host + local models", H.localData, async data => {
  applySnapshot({ host: { data }, nodes: {} });
  scratch.innerHTML = [...models_by_slug.values()].filter(m => data.models.some(x => x.slug === m.slug))
    .map(m => manageMarkup(m) + testsMarkup(m) + keyMarkup(m) + benchMarkup(m)).join("");
  renderAllModels();
});
applySnapshot({ host: { data: clone(H.localData) }, nodes: {} });

// ── คะแนน (ไฟล์ bench อยู่บน node) ──
await walk("benchmarks list", H.benchFleet, async d => { H.served.bench = d; await loadBench(); });
delete H.served.bench;
await walk("benchmark details", H.benchRun, async d => { H.served.benchRun = d; await showBenchDetail("full", "n1"); });
delete H.served.benchRun;
// ── weights บนดิสก์ (ผล scan ของ node) ──
await walk("weights scan", H.scan, async d => { H.served.scan = d; await loadScan(); });
delete H.served.scan;
// ── job: ผลของงานบน node และบนเครื่องนี้ ──
await walk("job on a node", H.job, async d => { H.served.job = d; await followNodeJob("n1", "full", "job-" + Math.random()); });
await walk("job on this machine", H.job, async d => { H.alerts.length = 0; H.served.job = d; await followJob("local-full", "job-" + Math.random()); });
// ── คลัสเตอร์ · ตาราง fit · แผน deploy · คำแนะนำ context · ผล build · สรุปฟลีต · consistency · สูตร ──
location.hash = "#/nodes"; await H.tick(6);
await walk("cluster view", H.cluster, async d => { H.served.cluster = d; await loadCluster(); });
delete H.served.cluster; await loadCluster();
await walk("fit table", H.fitPlan, async p => { scratch.innerHTML = fitTableHtml(p); });
await walk("deploy plan", H.plan, async d => { planView(d); await H.tick(1); });
planView(clone(H.plan)); await H.tick(1);
await walk("context advice", H.advice, async d => { H.served.advice = d; await paintContextAdvice(); });
delete H.served.advice;
await walk("build result", H.built, async r => { resultView(r); });
document.getElementById("w-close") && document.getElementById("w-close").click();
await walk("fleet summary tiles", H.summary, async s => { renderFleetSummary(s); renderRail(); });
location.hash = "#/overview"; await H.tick(6);
await walk("fleet consistency", H.consistency, async d => { lastFleetConsistency = d; renderOverview(true); });
await walk("recipes", H.recipes, async d => { H.served.recipes = d; await loadRecipes(); recipeList = d.recipes || []; wizForm(); pickRecipe(0); });
delete H.served.recipes;
document.getElementById("w-close") && document.getElementById("w-close").click();

// ── คำตอบที่ล้ม: ทุกรูปของ detail × ทุกจอที่โชว์ error ──
const shapes = { string: m => ({ detail: m }),
                 "choose-gguf": m => ({ detail: { kind: "choose-gguf", message: m, variants: [{ filename: m, size_gb: m }] } }),
                 "no-fit": m => ({ detail: { kind: "no-fit", message: m, alternatives: [m] } }),
                 gated: m => ({ detail: { kind: "gated", message: m } }),
                 object: m => ({ detail: { kind: m, message: m, alternatives: [m], variants: [{ filename: m, size_gb: m }], gates: [{ name: m, detail: m, passed: false }] } }),
                 "422 list": m => ({ detail: [{ loc: ["body", m], msg: m, type: m }] }), "not json": m => m };
const screens = {
  "scan": async body => { H.served.scan = { status: 500, body }; await loadScan(); delete H.served.scan; },
  "bench details": async body => { H.served.benchRun = { status: 500, body }; await showBenchDetail("full", "n1"); delete H.served.benchRun; },
  "doctor panel": async body => { H.served.localPanel = { status: 500, body }; await load("local-full", "doctor"); delete H.served.localPanel; },
  "key panel": async body => { H.served.localPanel = { status: 500, body };
     document.getElementById("panel-local-full").innerHTML = keyMarkup({ slug: "local-full" }); await loadKeyState("local-full"); delete H.served.localPanel; },
  "add machine": async body => { location.hash = "#/nodes"; await H.tick(2); document.getElementById("node-new").click();
     document.getElementById("n-host").value = "10.0.0.9"; H.served.addNode = { status: 500, body };
     document.getElementById("n-save").click(); await H.tick(4); delete H.served.addNode; },
  "deploy analyse": async body => { document.getElementById("new").click(); document.getElementById("w-model").value = "Org/Full";
     H.served.analyze = { status: 422, body }; await analyzeNow(); delete H.served.analyze; },
  "cluster": async body => { H.served.cluster = { status: 500, body }; await loadCluster(); delete H.served.cluster; },
};
const errors = { found: [], threw: [], cases: 0 };
for (const [shape, make] of Object.entries(shapes)) for (const [screen, show] of Object.entries(screens)) {
  errors.cases++;
  try { await show(make(MARK(`${screen} / ${shape}`))); } catch (err) { errors.threw.push(`${screen} / ${shape}: ${String(err && err.message || err).slice(0, 90)}`); }
  const hit = offenders();
  if (hit.length) errors.found.push(`${screen} / ${shape} → ${[...new Set(hit)].slice(0, 2).join(" | ")}`);
  scratch.innerHTML = "";
  document.getElementById("w-close")?.click();          // จอ error ของ wizard ต้องไม่ค้างไปปนกับเคสถัดไป
  if (document.getElementById("n-cancel")) { document.getElementById("n-cancel").click(); await H.tick(4); }
}
report["error responses"] = { leaves: errors.cases, found: errors.found, threw: errors.threw };
// ตัวควบคุม: marker เดียวกันที่ถูกแปะลง markup ตรง ๆ ต้องถูกจับได้ — ไม่งั้นเทสนี้ผ่านเพราะตาบอด ไม่ใช่เพราะหน้าปลอดภัย
scratch.innerHTML = `<span title='${esc(MARK("ctl-single-quote"))}'></span><i class=${esc(MARK("ctl-unquoted"))}></i>${MARK("ctl-text")}<b onclick="f('${esc(MARK("ctl-handler"))}')"></b>`;
const control = offenders().length;
scratch.innerHTML = "";
console.log(JSON.stringify({ report, control }));
H.errors.length = 0;
