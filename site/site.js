// RoboHub v2: one sentence -> auto-routed Runway Model Router -> physics gates -> SmolVLA films.
//   live:   the local server (GET /api/trainer answers): POST /api/footage, /api/data, /api/train, /api/vla, SSE per job
//   static: no server (Vercel): Train replays the recorded run (data/recorded.json), nothing is spent
//   ?static=1 forces the static replay.
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const fmt = (n) => Math.round(n).toLocaleString("en-US");
  const clock = (s) => { s = Math.max(0, Math.round(s)); const m = Math.floor(s / 60); return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}:${String(s % 60).padStart(2, "0")}`; };
  const qs = new URLSearchParams(location.search);
  // a new prompt (not a library example) loads for at least this long, and until its Runway clips settle, before any
  // MuJoCo preview appears
  const MIN_LOAD_MS = 30000;
  const FOOTAGE = 3; // Runway demonstration clips generated for every new prompt before its preview
  const FRAME_CREDITS = 5; // gen4_image first frame, 720p
  const DRY_FALLBACK = {
    "demo-cheap": { model: "veo-3.1-fast-generate-preview", provider: "google" },
    "demo-fast": { model: "veo-3.1-fast-generate-preview", provider: "google" },
    "demo-best": { model: "veo-3.1-generate-preview", provider: "google" },
    "robohub-q40": { model: "veo-3.1-fast-generate-preview", provider: "google" },
    "robohub-q80": { model: "veo-3.1-fast-generate-preview", provider: "google" },
    "robohub-q120": { model: "veo-3.1-generate-preview", provider: "google" },
    "robohub-q160": { model: "veo-3.1-generate-preview", provider: "google" },
    "robohub-q200": { model: "veo-3.1-generate-preview", provider: "google" },
  };

  const input = $("#prompt");
  const liveBtn = $("#live");
  const state = { live: false, backend: false, route: null, dry: { ...DRY_FALLBACK }, recorded: null, running: false };

  // ---------- backend or static ----------
  const getJSON = async (url, opt = {}, ms = 0) => {
    const ctl = new AbortController(); const t = ms ? setTimeout(() => ctl.abort(), ms) : 0;
    try {
      const r = await fetch(url, { ...opt, signal: ctl.signal });
      const ct = r.headers.get("content-type") || "";
      const body = ct.includes("json") ? await r.json() : null;
      return { ok: r.ok && body != null, status: r.status, body };
    } catch { return { ok: false, status: 0, body: null }; } finally { if (t) clearTimeout(t); }
  };
  const post = (url, data) => getJSON(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
  const ready = (async () => {
    const [dry, rec, probe] = await Promise.all([
      getJSON("data/router-dryruns.json"), getJSON("data/recorded.json"),
      qs.has("static") || /\.vercel\.app$/.test(location.hostname) ? Promise.resolve({ ok: false }) : getJSON("/api/trainer", {}, 1500),
    ]);
    if (dry.ok) for (const k of Object.keys(DRY_FALLBACK)) {
      const d = dry.body[k]; if (d && d.model) state.dry[k] = { model: d.model, provider: d.provider, credits: (d.estimatedCost || {}).credits ?? DRY_FALLBACK[k].credits };
    }
    state.recorded = rec.ok ? rec.body : null;
    state.backend = !!probe.ok;
  })();

  // ---------- auto-route (debounced) ----------
  const pill = $("#route-pill"), line = $("#route-line");
  let routeSeq = 0, routeTimer = 0;
  const textNow = () => input.value.trim() || input.placeholder;
  function paintRoute(r, dry, note) {
    const name = Route.NAME[r.router] || r.router;
    if (!state.route || state.route.router !== r.router) { pill.classList.remove("flash"); void pill.offsetWidth; pill.classList.add("flash"); }
    state.route = { ...r, dry };
    $("#route-label").textContent = name;
    line.classList.toggle("is-no", !!note);
    line.innerHTML = note ? esc(note)
      : `<b>${esc(name)}</b> a run: Gemini <b>${esc(dry.model || "Veo")}</b> · ${FOOTAGE} clips on your key`;
  }
  async function route() {
    const seq = ++routeSeq;
    if (!input.value.trim()) { pill.classList.add("is-empty"); line.textContent = ""; state.route = null; return; }
    pill.classList.remove("is-empty");
    const text = textNow();
    const local = Route.decide(text, { budget: Route.budget });
    paintRoute(local, state.dry[local.router] || DRY_FALLBACK[local.router]);
    await ready;
    if (!state.backend || true) return; // the budget picks the router; the dry-run table above is the live pick
    const r = await post("/api/route", { task: text, live: state.live });
    if (seq !== routeSeq || !r.ok || !r.body.router) return;
    const d = r.body.dryRun || r.body.dry_run || {};
    const base = state.dry[r.body.router] || DRY_FALLBACK[r.body.router] || {};
    paintRoute({ router: r.body.router, reasons: r.body.reasons && r.body.reasons.length ? r.body.reasons : local.reasons },
      { model: d.model || base.model, provider: d.provider || base.provider, credits: d.credits ?? d.estimated_credits ?? base.credits });
  }
  const routeSoon = () => { clearTimeout(routeTimer); routeTimer = setTimeout(route, 250); };
  input.addEventListener("input", routeSoon);
  // budget picker: what one run may spend on Runway ($2 to $10); every option is a quality router with that ceiling
  const budSeg = $("#budget-seg");
  for (const b of Route.BUDGETS) {
    const btn = el("button", null, `$${b}`); btn.type = "button"; btn.setAttribute("role", "radio");
    btn.addEventListener("click", () => { Route.budget = b; paintBudget(); route(); });
    budSeg.append(btn);
  }
  function paintBudget() { [...budSeg.children].forEach((n, i) => { const on = Route.BUDGETS[i] === Route.budget; n.classList.toggle("on", on); n.setAttribute("aria-checked", String(on)); }); }
  paintBudget();
  document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => { input.value = c.dataset.text; input.focus(); route(); }));
  route();

  // ---------- GPU for SmolVLA training (RunPod on-demand, prices seen 2026-09-30) ----------
  const GPUS = [
    { key: "4090", label: "4090", id: "NVIDIA GeForce RTX 4090", price: "$0.34/h" },
    { key: "5090", label: "5090", id: "NVIDIA GeForce RTX 5090", price: "$0.69/h", speed: "3,000 steps in 9 min" },
    { key: "l40s", label: "L40S", id: "NVIDIA L40S", price: "$0.79/h" },
    { key: "a100", label: "A100 80GB", id: "NVIDIA A100 80GB PCIe", price: "$1.19/h" },
    { key: "h100", label: "H100", id: "NVIDIA H100 80GB HBM3", price: "price varies" },
  ];
  state.gpu = GPUS[1];
  state.runTimes = {};

  // ---------- run time (the Time card) ----------
  const M = { t0: 0, seconds: null, timer: 0 };
  const paintTime = () => { $('.stats [data-m="time"] b').textContent = clock(M.seconds != null ? M.seconds : M.t0 ? Date.now() / 1000 - M.t0 : 0); };
  function timeStart() { Object.assign(M, { t0: Date.now() / 1000, seconds: null }); clearInterval(M.timer); M.timer = setInterval(paintTime, 500); paintTime(); }
  function timeStop() { if (M.seconds == null && M.t0) M.seconds = Date.now() / 1000 - M.t0; clearInterval(M.timer); paintTime(); if (CUR.slug) state.runTimes[CUR.slug] = M.seconds; }
  const gpuSeg = $("#gpu-seg"), gpuHint = $("#gpu-hint");
  function paintGpu() {
    gpuSeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.gpu === state.gpu.key)));
    gpuHint.textContent = state.gpu.speed ? `${state.gpu.price} · ${state.gpu.speed}` : state.gpu.price;
  }
  for (const g of GPUS) {
    const b = el("button", null, esc(g.label)); b.type = "button"; b.dataset.gpu = g.key; b.setAttribute("role", "radio"); b.title = `${g.id} · ${g.price}`;
    b.addEventListener("click", () => { state.gpu = g; paintGpu(); });
    gpuSeg.append(b);
  }
  paintGpu();

  // ---------- run head + tiles ----------
  // running-job text lives as a tiny label on this run's unfinished tiles; cleared when the run ends
  const status = (t, done = false) => document.querySelectorAll(`#grid .tile[data-run="${RUN}"]`).forEach((n) => {
    const s = $(".tile-run", n); s.textContent = done ? "" : t; s.hidden = done || !t;
  });
  const tiles = new Map();
  let RUN = 0, CUR = { title: "", router: "", slug: "" };
  const STATE = {
    queued: ["wait", "Queued"], gen: ["", "Runway generating"], physics: ["", "Physics check"], pv: ["ok", "Scripted preview"],
    ok: ["ok", "Accepted"], no: ["no", "Rejected"], fail: ["no", "Runway failed"], train: ["", "SmolVLA training"], film: ["", "SmolVLA filming"],
  };
  function tile(id, grid = $("#grid"), key = `${RUN}:${id}`, meta = null) {
    if (tiles.has(key)) return tiles.get(key);
    const root = el("article", "tile");
    const m = meta || CUR;
    root.dataset.run = meta ? "lib" : String(RUN);
    if (m.slug) root.dataset.slug = m.slug;
    root.innerHTML = `<span class="tile-id"><span class="route-pill xs">${pillHTML(m.router)}</span><span class="tile-title">${esc(m.title || id)}</span></span><video muted loop playsinline preload="metadata"></video>
      <div class="tile-state"><span class="status wait"><i></i><span>Queued</span></span><small></small></div>
      <span class="tile-cap"></span><span class="tile-time" hidden></span><span class="tile-run" hidden></span>`;
    const older = meta ? null : [...grid.children].find((c) => c.dataset.run !== String(RUN));
    if (older) grid.insertBefore(root, older); else grid.append(root);
    const t = {
      root, src: null,
      set(k, note = "", text = null) {
        const [cls, label0] = STATE[k], label = text || label0;
        if (k === "no" && t.sim) { t.unpreview(); root.classList.remove("is-film"); }
        const s = $(".status", root); s.className = `status ${cls}`; $("span", s).textContent = label;
        $("small", root).textContent = note; root.classList.toggle("is-no", k === "no" || k === "fail"); return t;
      },
      time(sec) { if (sec == null || !isFinite(sec)) return t; const b = $(".tile-time", root); b.textContent = clock(sec); b.title = "Time for this run"; b.hidden = false; return t; },
      film(src, success = true, cap = null, cycling = false) {
        const v = $("video", root); t.src = src; t.real = true;
        t.unpreview();
        if (!cycling) { v.onended = null; root.classList.remove("is-clips"); }
        v.onerror = () => { root.classList.remove("is-film"); t.set("film", "film not found"); };
        v.muted = true; v.loop = true; v.autoplay = true; v.playsInline = true;
        v.src = src; v.play().catch(() => {}); v.oncanplay = () => { if (v.paused) v.play().catch(() => {}); };
        const c = $(".tile-cap", root); c.textContent = cap || `SmolVLA · ${success ? "in the bowl" : "missed"}`; c.classList.toggle("miss", !success);
        c.title = "";
        root.classList.add("is-film"); return t;
      },
      // a new prompt's Runway clips (t.runway[i] = {url, model, k}) play in this one tile in turn: 1 -> 2 -> 3 -> 1
      playlist() {
        const list = () => t.runway.filter(Boolean);
        if (!list().length || t.sim) return t;
        const v = $("video", root);
        const show = () => {
          const l = list(), it = l[t.pl % l.length];
          if (t.src === it.url && v.src) { v.currentTime = 0; v.play().catch(() => {}); return; }
          t.film(it.url, true, `Runway · ${it.model} · clip ${it.k}/${t.runwayN}`, true); v.loop = false;
        };
        root.classList.add("is-clips", "has-clips");
        v.onended = () => { t.pl = (t.pl + 1) % list().length; show(); };
        if (t.pl == null) { t.pl = 0; show(); }
        return t;
      },
      // the live in-browser MuJoCo preview (preview.js): scripted motion, never a learned policy
      preview(sim) {
        const v = $("video", root); v.onended = null; v.pause(); root.classList.remove("is-clips");
        t.sim = sim; root.dataset.preview = "1"; root.classList.add("is-film", "is-preview");
        root.insertBefore(sim.canvas, $(".tile-state", root));
        const c = $(".tile-cap", root); c.textContent = PREVIEW_CAP; c.classList.remove("miss");
        c.title = `Scripted motion, not a learned policy: ${sim.summary}${sim.note ? ` (${sim.note})` : ""}`;
        root.title = c.title;
        sim.ready.catch(() => {
          if (t.sim !== sim) return;
          t.unpreview();
          if (t.runway && t.runway.some(Boolean)) { t.pl = null; t.playlist(); t.set("pv", "", "Preview unavailable"); return; }
          root.classList.remove("is-film"); t.set(t.real ? "film" : "queued", "preview unavailable in this browser");
        });
        return t;
      },
      cap(text, miss = false) { const c = $(".tile-cap", root); c.textContent = text; c.classList.toggle("miss", miss); return t; },
      unpreview() {
        if (!t.sim) return;
        t.sim.stop(); t.sim = null; delete root.dataset.preview; root.classList.remove("is-preview"); root.removeAttribute("title");
      },
    };
    t.key = key;
    root.addEventListener("click", () => {
      if (root.classList.contains("is-film") && !root.dataset.live && !root.dataset.preview) openOverview(root.dataset.slug, t.src, m);
      else if (t.runway && t.runway.some(Boolean)) openClips(t, m); // a new prompt's Runway clips, in the overview gallery
    });
    tiles.set(key, t);
    return t;
  }
  const pillHTML = (router) => `<span class="robot-dot"></span><span>${esc(Route.NAME[router] || router || "")}</span>`;
  function startRun(task, head, router) {
    if (state.pvRun === RUN) state.pvRun = null; else RUN++;
    CUR = { title: task, router, slug: "" };
    $("#fleet").hidden = false; $("#run-task").textContent = task; $("#run-pill").innerHTML = pillHTML(router); status(head); timeStart();
    $("#stats").hidden = false; $("#run-pill").hidden = false;
    requestAnimationFrame(() => window.scrollTo({ top: $("#fleet").getBoundingClientRect().top + scrollY - 12, behavior: "smooth" }));
  }

  // ---------- overview: the film plus how its VLA was trained (data/overview.json, keyed by slug) ----------
  //   the film's own JSON (<clip>.json) overrides eval {successes, seeds} and download when present
  const sheet = $("#sheet"), sv = $("#sheet-video"), steps = $("#ov-steps"), dl = $("#ov-dl");
  let overview = null, ovSeq = 0;
  const minutes = (txt) => { const m = String(txt || "").match(/([\d.]+)\s*min/); return m ? Number(m[1]) * 60 : null; };
  const loadOverview = () => (overview ||= getJSON(`data/overview.json?t=${Date.now()}`).then((r) => (r.ok ? r.body : {})));
  const num = (n) => (n == null || n === "" ? "" : typeof n === "number" ? fmt(n) : esc(n));
  const join = (...xs) => xs.filter((x) => x != null && x !== "").join(" · ");
  function renderOverview(o, meta, film) {
    const r = o.route || {}, d = o.demonstrations || {}, ds = o.dataset || {}, t = o.training || {};
    const ev = (film && film.eval && film.eval.seeds ? film.eval : null) || (o.result && o.result.seeds ? o.result : null);
    const secs = state.runTimes[meta.slug] ?? o.run_seconds ?? null;
    const url = (film && film.download) || (o.download && o.download.url) || "";
    const row = (k, main, sub = "") => `<li><span class="ov-k">${k}</span><p class="ov-v">${main}</p>${sub ? `<p class="ov-s">${sub}</p>` : ""}</li>`;
    steps.innerHTML = [
      row("Prompt", `<q>${esc(o.prompt || meta.title || "")}</q>`),
      row("Route", join(`<b>${esc(r.router || Route.NAME[meta.router] || "")}</b> router`, esc(r.model), r.credits_per_clip != null ? `${num(r.credits_per_clip)} credits a clip` : ""), esc(r.note || "")),
      row("Demonstrations", d.generated != null ? `${num(d.generated)} clips generated, <b>${num(d.accepted)}</b> physics-accepted` : `<b>${num(d.episodes)}</b> episodes`,
        join(esc(d.rejected || ""), esc(d.source || ""))),
      row("Dataset", join(ds.episodes != null ? `<b>${num(ds.episodes)}</b> episodes` : "", ds.frames != null ? `${num(ds.frames)} frames` : ""), ds.cameras ? `cameras ${esc(ds.cameras)}` : ""),
      row("Training", join(`${esc(t.base || "SmolVLA")} base`, t.steps != null ? `<b>${num(t.steps)}</b> steps` : ""), join(esc(t.gpu || ""), esc(t.time || ""), esc(t.cost || ""))),
      row("Time", join(secs != null ? `<b>${clock(secs)}</b> demo pipeline` : "", t.time ? `${esc(t.time)} training` : "") || `<span class="ov-pending">not recorded yet</span>`,
        secs != null ? "sentence to physics-accepted dataset, then SmolVLA training" : ""),
      row("Result", ev ? `<b class="ov-score">${num(ev.successes)}/${num(ev.seeds)}</b>` : `<span class="ov-pending">Evaluating</span>`,
        ev ? esc((o.result && o.result.note) || "in simulation") : "score in simulation"),
    ].join("");
    if (url) { dl.href = url; dl.removeAttribute("aria-disabled"); dl.innerHTML = `Download VLA${o.download && o.download.size && !(film && film.download) ? ` <small>${esc(o.download.size)}</small>` : ""}`; }
    else { dl.removeAttribute("href"); dl.setAttribute("aria-disabled", "true"); dl.innerHTML = "Download VLA <small>uploading</small>"; }
  }
  // ---------- gallery: every Runway clip for the task (data/gallery.json, keyed by slug) ----------
  //   [{id, url, router, model, credits, verdict: "Accepted" | "Rejected" | null, reason}]
  const gal = $("#ov-gallery"), grid = $("#ov-grid"), gsum = $("#ov-gsum"), jump = $("#ov-jump");
  let gallery = null;
  const loadGallery = () => (gallery ||= getJSON(`data/gallery.json?t=${Date.now()}`).then((r) => (r.ok ? r.body : {})));
  const lazy = "IntersectionObserver" in window ? new IntersectionObserver((es) => {
    for (const e of es) {
      const v = e.target;
      if (e.isIntersecting) { if (!v.src && v.dataset.src) v.src = v.dataset.src; v.play().catch(() => {}); }
      else if (v.src) v.pause();
    }
  }, { rootMargin: "200px 0px" }) : null;
  function clearGallery() {
    for (const v of grid.querySelectorAll("video")) { lazy && lazy.unobserve(v); v.pause(); v.removeAttribute("src"); v.load(); }
    grid.innerHTML = ""; gal.hidden = true; jump.hidden = true;
  }
  function renderGallery(clips) {
    clearGallery();
    if (!clips || !clips.length) return;
    const credits = clips.reduce((s, c) => s + (c.credits || 0), 0);
    const judged = clips.filter((c) => c.verdict), ok = judged.filter((c) => c.verdict === "Accepted").length;
    const models = [...new Set(clips.map((c) => c.model).filter(Boolean))];
    gsum.textContent = join(`${clips.length} clips`, `${fmt(credits)} credits`, models.join(", "), judged.length ? `${ok}/${judged.length} judged clips accepted` : "");
    for (const c of clips) {
      const li = el("li", "ov-clip");
      const v = el("video");
      v.muted = true; v.loop = true; v.playsInline = true; v.preload = "none";
      v.setAttribute("muted", ""); v.setAttribute("playsinline", ""); v.setAttribute("aria-label", `Runway clip ${c.id}`);
      if (lazy) { v.dataset.src = c.url; lazy.observe(v); } else { v.src = c.url; v.autoplay = true; }
      const vd = c.verdict ? `<span class="ov-vd ${c.verdict === "Accepted" ? "is-ok" : "is-no"}">${esc(c.verdict)}</span>` : "";
      const why = c.verdict === "Rejected" && c.reason ? `<br><span class="ov-why">${esc(c.reason)}</span>` : "";
      li.append(v, el("p", "", `${vd}<b>${esc(c.id)}</b> · ${esc(c.router || "?")} · ${esc(c.model || "?")}${c.credits != null ? ` · ${fmt(c.credits)} cr` : ""}${why}`));
      grid.append(li);
    }
    jump.textContent = `See all ${clips.length} Runway clips below`;
    gal.hidden = false; jump.hidden = false;
  }
  jump.addEventListener("click", (e) => { e.preventDefault(); gal.scrollIntoView({ behavior: "smooth", block: "start" }); });

  async function openOverview(slug, src, meta) {
    const seq = ++ovSeq; dl.hidden = false;
    sv.src = src; sheet.hidden = false; document.body.classList.add("is-locked"); sv.play().catch(() => {});
    sheet.scrollTop = 0; $(".sheet-panel.ov").scrollTop = 0;
    loadGallery().then((g) => { if (seq === ovSeq) renderGallery(g[slug]); });
    const all = await loadOverview();
    if (seq !== ovSeq) return;
    const o = all[slug] || {};
    renderOverview(o, meta, null);
    const j = await getJSON(String(src).replace(/\.mp4(\?.*)?$/, ".json"));
    if (seq === ovSeq && j.ok && (j.body.eval || j.body.download)) renderOverview(o, meta, j.body);
  }
  function openClips(t, meta) {
    ++ovSeq;
    const cl = t.runway.filter(Boolean), router = Route.NAME[meta.router] || meta.router || "";
    sv.src = cl[0].url; sheet.hidden = false; document.body.classList.add("is-locked"); sv.play().catch(() => {});
    sheet.scrollTop = 0; $(".sheet-panel.ov").scrollTop = 0;
    const models = [...new Set(cl.map((c) => c.model).filter(Boolean))].join(", ");
    steps.innerHTML = `<li><span class="ov-k">Prompt</span><p class="ov-v"><q>${esc(meta.title)}</q></p></li>
      <li><span class="ov-k">Demonstrations</span><p class="ov-v"><b>${cl.length}</b> of ${t.runwayN} Runway clips, generated live</p><p class="ov-s">${esc(join(`${router} budget`, models))}</p></li>`;
    dl.hidden = true;
    renderGallery(cl.map((c) => ({ id: `clip ${c.k}`, url: c.url, router, model: c.model, credits: c.credits ?? null, verdict: c.verdict || null })));
  }
  function closeFilm() { ovSeq++; clearGallery(); sheet.hidden = true; sv.pause(); sv.removeAttribute("src"); sv.load(); document.body.classList.remove("is-locked"); }
  $("#sheet-close").addEventListener("click", closeFilm);
  $("#sheet-backdrop").addEventListener("click", closeFilm);
  dl.addEventListener("click", (e) => { if (dl.getAttribute("aria-disabled") === "true") e.preventDefault(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !sheet.hidden) closeFilm(); });

  // ---------- one row per prompt: data/three-prompts.json
  //   [{prompt, slug, router, model, credits, clips, accepted, episodes, frames, seconds?, eval: {successes, seeds}, tiles_dir}]
  async function manifest() {
    for (const u of ["data/three-prompts.json"]) { // the server has no /api/three-prompts; both modes read the built file
      const r = await getJSON(`${u}?t=${Date.now()}`);
      if (r.ok) return Array.isArray(r.body) ? r.body : Array.isArray(r.body.prompts) ? r.body.prompts : [];
    }
    return [];
  }
  const score = (e) => (e && e.seeds ? `SmolVLA <b>${e.successes}/${e.seeds}</b>` : "");
  const count = (x) => (Array.isArray(x) ? x.length : x);
  async function prompts(mainSlug, base) {
    const rows = await manifest();
    for (const e of rows) {
      if (!e || !e.slug || document.querySelector(`#grid .tile[data-slug="${e.slug}"]`)) continue;
      const dir = String(e.tiles_dir || "media/vla").replace(/^.*\/web-runs\/[^/]+\//, "").replace(/^\/+|\/+$/g, "");
      let ids = Array.isArray(e.accepted) ? e.accepted : Array.isArray(e.films) ? e.films : [];
      if (!ids.length && state.backend) { const v = await getJSON(`/api/runs/${e.slug}/vla`); if (v.ok) ids = Object.keys(v.body.films || {}); }
      if (!ids.length) continue;
      const id = String(ids[0]).replace(/\.mp4$/, ""), url = `${base}/${e.slug}/${dir}/${id}.mp4`;
      const t = tile(id, $("#grid"), `lib:${e.slug}`, { title: e.prompt || e.slug, router: e.router, slug: e.slug }).film(url, true, " ");
      loadOverview().then((o) => { const v = o[e.slug] || {}; t.time(v.run_seconds ?? minutes((v.training || {}).time)); });
      getJSON(url.replace(/\.mp4$/, ".json")).then((j) => {
        const c = $(".tile-cap", t.root);
        // a preview film is the scripted operator doing the task, not a trained VLA: say so
        if (j.ok && j.body.source === "preview") { c.textContent = "Training still in progress"; return; }
        if (!j.ok || j.body.success == null) { c.textContent = "SmolVLA"; return; }
        c.textContent = `SmolVLA · ${j.body.success ? "success" : "missed"}`; c.classList.toggle("miss", !j.body.success);
      });
    }
  }


  // ---------- static: replay the recorded run ----------
  async function replay(_text, fast = false) {
    const R = state.recorded;
    const wait_ = (ms) => (fast ? Promise.resolve() : wait(ms));
    if (!R) { line.classList.add("is-no"); line.textContent = "No recorded run on this page."; return; }
    const tag = `Recorded run · ${R.model} · no credits spent`;
    // static site: a typed push / stack prompt replays that recorded example; anything else replays the pick run
    const typed = String(_text || "").toLowerCase();
    const want = /\bpush|shove|slide\b/.test(typed) ? "push" : /\b(tower|three blocks)\b/.test(typed) ? "stack-three" : /\bstack|tower|on top\b/.test(typed) ? "stack" : null;
    const lib = want && !fast ? (await manifest()).find((e) => e && e.slug && e.slug.startsWith(want)) : null;
    startRun(lib ? lib.prompt : R.task, tag, lib ? lib.router : R.router);
    // the page-load preload only fills the library: no run happened, so no run title, route or time
    if (fast) { $("#run-task").textContent = "Library"; $("#run-pill").hidden = true; $("#stats").hidden = true; }
    const filmDir = lib ? `runs/${lib.slug}/${lib.tiles_dir || "media/vla"}` : `runs/${R.slug}/media/vla`;
    const hero = lib ? { id: lib.accepted[0], film: `${lib.accepted[0]}.mp4`, success: true }
      : R.clips.find((c) => c.accepted && c.film && c.success !== false) || R.clips.find((c) => c.film);
    CUR.slug = lib ? lib.slug : R.slug;
    document.querySelectorAll(`#grid .tile[data-slug="${CUR.slug}"]`).forEach((n) => n.remove());
    const t = tile(hero ? hero.id : "demo");
    if (!fast) attachPreview(t);
    const T = fast ? 0 : 9000, t0 = performance.now(), myRun = RUN;
    M.t0 = 0; M.seconds = 0; clearInterval(M.timer);
    const tick = () => { if (RUN !== myRun) return; const f = T ? Math.min(1, (performance.now() - t0) / T) : 1; M.seconds = R.seconds * f; paintTime(); if (f < 1 && state.running) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
    status("Gemini generating · recorded run"); t.set("gen"); await wait_(1600);
    status("Physics check · 21 gates in MuJoCo"); t.set("physics"); await wait_(1800);
    t.set("ok"); await wait_(700);
    status(`SmolVLA training on ${R.episodes} episodes`); t.set("train");
    await wait_(Math.max(0, T - (performance.now() - t0) - 600));
    if (hero) t.film(`${filmDir}/${hero.film}`, hero.success !== false, lib && !lib.slug.includes("bowl") ? "SmolVLA · success" : null);
    if (RUN === myRun) { M.seconds = R.seconds; paintTime(); } state.runTimes[R.slug] = R.seconds;
    t.time(lib ? null : R.seconds);
    status("", true);
    prompts(R.slug, "runs");
  }

  // ---------- cloud: a new sentence is generated live by Runway (Vercel functions in api/cloud) ----------
  //   a sentence that matches a finished task replays it; anything else: route -> gen4_image first frame -> Model Router video
  // bring your own key: on the hosted site new prompts run on the visitor's Runway credits; the key stays in this browser
  const KEY = "robohub.geminiKey", keyRow = $("#key-row"), keyIn = $("#rw-key");
  try { keyIn.value = localStorage.getItem(KEY) || ""; } catch {}
  keyIn.addEventListener("input", () => { keyRow.classList.remove("need"); try { localStorage.setItem(KEY, keyIn.value.trim()); } catch {} });
  ready.then(() => { keyRow.hidden = state.backend; });
  const withKey = (opt = {}) => ({ ...opt, headers: { ...(opt.headers || {}), "X-Gemini-Key": keyIn.value.trim() } });
  const cpost = (url, data) => getJSON(url, withKey({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) }));
  const norm = (s) => String(s || "").toLowerCase().replace(/\s+/g, " ").trim().replace(/[.!]+$/, "");
  let libSet = null;
  const isLibrary = async (text) => {
    libSet ||= manifest().then((rows) => new Set([...document.querySelectorAll(".chip")].map((c) => c.dataset.text)
      .concat(rows.map((e) => e && e.prompt), state.recorded ? [state.recorded.task] : []).filter(Boolean).map(norm)));
    return (await libSet).has(norm(text));
  };
  const libraryMatch = (text) => /\b(bowl|push|shove|slide|stack|tower|on top)\b/i.test(String(text || ""));
  async function poll(id, onTick) {
    for (let i = 0; i < 200; i++) {
      await wait(i < 3 ? 2500 : 4000);
      const r = await getJSON(`/api/cloud/task?id=${encodeURIComponent(id)}`, withKey());
      if (r.status === 401) throw new Error((r.body && r.body.error) || "Gemini rejected this API key.");
      if (!r.ok) continue;
      if (r.body.status === "SUCCEEDED") return r.body.output && r.body.output[0];
      if (r.body.status === "FAILED" || r.body.status === "CANCELLED") throw new Error(r.body.failure || "Gemini could not make this clip.");
      onTick(r.body);
    }
    throw new Error("Gemini is taking too long; try again.");
  }
  const errOf = (r, fallback) => (r.body && (r.body.error || r.body.detail || r.body.reason)) || fallback;

  // ---------- a new prompt gets ONE tile: Runway progress in its status, finished clips play in it in turn, and the
  //   MuJoCo preview replaces them once MIN_LOAD_MS has passed and every clip has finished or failed ----------
  //   runTile(n, keyless, T) -> R: R.clip(i, {frame, video, end}), R.add(i, url, model, credits), R.gate(p), R.stop()
  function runTile(n, keyless = false, T = tile("run")) {
    const t0 = performance.now();
    T.root.dataset.live = "1"; T.runway = []; T.runwayN = n;
    const C = Array.from({ length: n }, () => ({ frame: 0, video: 0, end: null }));
    let stopped = false;
    const R = {
      T,
      elapsed: () => (performance.now() - t0) / 1000,
      ended: (i) => !!(C[i] && C[i].end),
      clip(i, patch) { if (!C[i] || C[i].end) return; Object.assign(C[i], patch); paint(); },
      add(i, url, model, credits = null) {
        if (!C[i] || C[i].end) return;
        Object.assign(C[i], { frame: 1, video: 1, end: "ok" });
        T.runway[i] = { url, model, credits, k: i + 1 };
        T.playlist(); paint();
      },
      // resolves with p's value once MIN_LOAD_MS has passed and p has settled
      gate: (p) => Promise.all([wait(Math.max(0, MIN_LOAD_MS - (performance.now() - t0))), p]).then(([, v]) => v),
      stop() { stopped = true; clearInterval(timer); },
    };
    function paint() {
      if (stopped) return;
      const left = Math.max(0, Math.ceil(MIN_LOAD_MS / 1000 - R.elapsed()));
      const ok = C.filter((c) => c.end === "ok").length, bad = C.filter((c) => c.end === "fail").length;
      if (keyless) { T.set("gen", `no Gemini key · preview in ${left} s`, "Preparing preview"); return; }
      if (ok + bad < n) {
        const pct = Math.round((100 * C.reduce((s, c) => s + (c.end ? 1 : 0.25 * Math.min(1, c.frame) + 0.75 * Math.min(1, c.video)), 0)) / n);
        const framing = C.some((c) => !c.end && c.frame < 1);
        T.set("gen", `${framing ? "first frames" : "clip videos"} · ${ok}/${n} done · ${pct}%${bad ? ` · ${bad} failed` : ""}`,
          ok ? `Gemini · ${ok}/${n} done · ${pct}%` : "Gemini generating");
      } else {
        T.set("gen", `${ok}/${n} Gemini clips${bad ? ` · ${bad} failed` : ""} · preview in ${left} s`, left ? `MuJoCo preview in ${left} s` : "Building MuJoCo preview");
      }
    }
    const timer = setInterval(paint, 500); paint();
    return R;
  }
  // the gated preview in the run's one tile
  async function showPreview(text, R, extra = "") {
    R.stop();
    const T = R.T; state.pvTile = T;
    if (!window.RoboHubPreview) { T.set("queued", "preview unavailable"); return T; }
    const sim = RoboHubPreview.mount(text);
    T.set("pv").preview(sim).time(R.elapsed());
    if (extra) T.cap(`${PREVIEW_CAP} · ${extra}`);
    const got = T.runway.filter(Boolean).length;
    if (got) T.root.title += ` · Click for the ${got} Runway clip${got > 1 ? "s" : ""}`;
    await Promise.race([sim.ready.catch(() => {}), wait(20000)]);
    return T;
  }

  // one Runway clip through the hosted (or local) /api/cloud: first frame (scene variant i) -> budget router video
  async function cloudClip(text, i, R, dry) {
    try {
      const s = await cpost("/api/cloud/start", { prompt: text, budget: Route.budget, variant: i });
      if (!s.ok) { if (s.status === 401) keyRow.classList.add("need"); throw new Error(errOf(s, "Live generation is unavailable right now.")); }
      R.clip(i, { frame: 1 });
      const v = await cpost("/api/cloud/video", { prompt: s.body.task, image_b64: s.body.image_b64, mime: s.body.mime, budget: Route.budget });
      if (!v.ok) throw new Error(errOf(v, "Veo refused this clip."));
      const model = v.body.model || dry.model;
      const uri = await poll(v.body.video_task, (b) => R.clip(i, { video: Math.min(0.95, b.progress || 0) }));
      const file = await fetch(`/api/cloud/media?uri=${encodeURIComponent(uri)}`, withKey());
      if (!file.ok) throw new Error("Could not download the Gemini clip.");
      const url = URL.createObjectURL(await file.blob());
      R.add(i, url, model, v.body.credits ?? null);
      return { ok: true, url, model };
    } catch (e) {
      R.clip(i, { end: "fail" });
      return { ok: false, error: String(e.message || e) };
    }
  }

  async function cloud(text) {
    const instant = !!state.pvTile; // a library sentence already has its preview up: no gate
    const rt = Route.decide(text, { budget: Route.budget });
    const dry = state.dry[rt.router] || DRY_FALLBACK[rt.router];
    const name = Route.NAME[rt.router] || rt.router;
    const hasKey = state.backend || !!keyIn.value.trim();
    startRun(text, "", rt.router); CUR.slug = "";
    if (!hasKey) {
      const R = runTile(0, true);
      keyRow.hidden = false; keyRow.classList.add("need");
      line.classList.add("is-no");
      line.textContent = `The ${FOOTAGE} Gemini clips need a Gemini API key: add yours below. The scripted MuJoCo preview is made in your browser.`;
      await R.gate(Promise.resolve());
      await showPreview(text, R, "no Runway footage: add a key");
      timeStop(); status("", true);
      return;
    }
    const R = runTile(FOOTAGE, false, instant ? state.pvTile : undefined);
    status(`Gemini generating · ${FOOTAGE} clips · ${name} budget`);
    const all = Promise.all(Array.from({ length: FOOTAGE }, (_, i) => cloudClip(text, i, R, dry)));
    const res = instant ? await all : await R.gate(all);
    const ok = res.filter((r) => r.ok);
    if (instant) { R.stop(); R.T.set("pv"); } else await showPreview(text, R, ok.length ? "" : "no Runway footage");
    timeStop(); status("", true);
    const bad = res.filter((r) => !r.ok);
    line.classList.toggle("is-no", !ok.length);
    line.textContent = !ok.length ? bad[0].error
      : `${ok.length} of ${FOOTAGE} live Runway clips, generated just now${bad.length ? ` (${bad.length} failed: ${short(bad[0].error, 60)})` : ""}. ` + (state.backend
        ? "The physics gates, retargeting and SmolVLA training cover pick-and-place tasks today, so this task gets its demonstration videos."
        : "The physics check, retargeting and SmolVLA training run in the full pipeline on a GPU (see the repo).");
  }

  // ---------- live: the real pipeline ----------
  const job = (id, on) => new Promise((resolve) => {
    const es = new EventSource(`/api/jobs/${encodeURIComponent(id)}/events`);
    let ok = false;
    es.onmessage = (m) => { let ev; try { ev = JSON.parse(m.data); } catch { return; } if (ev.type === "job_end") { ok = !!ev.ok; es.close(); resolve(ok); return; } on(ev); };
    es.onerror = () => { if (es.readyState === EventSource.CLOSED) resolve(ok); };
  });
  const why = (r, fallback) => (r.body && (r.body.reason || r.body.detail)) || fallback;
  const short = (s, n = 42) => (String(s).length > n ? String(s).slice(0, n - 1) + "…" : String(s));

  async function live(text, lib = false) {
    const rt = state.route || Route.decide(text, { budget: Route.budget });
    const dry = rt.dry || state.dry[rt.router] || DRY_FALLBACK[rt.router];
    const name = Route.NAME[rt.router] || rt.router;
    // a new sentence asks for FOOTAGE clips; a library sentence keeps the server's default run size
    const f = await post("/api/footage", lib ? { task: text, router: rt.router } : { task: text, router: rt.router, clips: FOOTAGE });
    if (!f.ok && f.body && f.body.infeasible) return cloud(text); // outside pick-and-place: live Runway clips instead
    if (!f.ok) {
      const msg = why(f, f.status === 409 ? "A stage is still running." : "The server refused this task.");
      paintRoute(rt, dry, msg);
      if (!lib) { // no footage was made; the preview still waits out the minimum
        const R = runTile(FOOTAGE);
        for (let i = 0; i < FOOTAGE; i++) R.clip(i, { end: "fail" });
        await R.gate(Promise.resolve()); await showPreview(text, R, "no Runway footage");
      }
      timeStop(); status("", true); return;
    }
    const { slug, clip_ids: ids = [], existing = [] } = f.body;
    let model = (f.body.estimate && f.body.estimate.model) || dry.model;
    startRun(text, `Live · ${model || name}`, rt.router); CUR.slug = slug;
    const base = "/runs";
    // a library sentence: one tile per scenario, and the ones it already had keep their films
    if (lib && existing.length) {
      const v = await getJSON(`/api/runs/${slug}/vla`);
      for (const c of existing) {
        const film = v.ok && v.body.films[c];
        if (film) tile(c).film(`${base}/${slug}/media/vla/${c}.mp4`, film.success !== false);
      }
    }
    // a new sentence: ONE tile for the whole run (runTile); every stage below reports on it
    const R = lib ? null : runTile(ids.length);
    const tt = (c) => (lib ? tile(c) : R.T);
    const idx = new Map(ids.map((c, i) => [c, i]));
    if (lib) { ids.forEach((c) => tile(c)); if (ids.length && !existing.length) attachPreview(tile(ids[0])); }
    status(`Runway generating · ${ids.length} clips · ${name} budget`);
    const okFoot = await job(f.body.job, (ev) => {
      const i = idx.get(ev.clip);
      if (ev.type === "runway" && ev.clip) {
        if (lib) tile(ev.clip).set("gen", ev.stage === "frame" ? "first frame" : "video");
        else if (i != null) { const p = ev.progress != null ? Math.min(0.95, ev.progress) : 0.05; R.clip(i, ev.stage === "frame" ? { frame: p } : { frame: 1, video: p }); }
      } else if (ev.type === "frame_ready" && !lib && i != null) R.clip(i, { frame: 1 });
      else if (ev.type === "routed" && ev.model) model = ev.model;
      else if (ev.type === "clip_ready") {
        if (lib || !ev.file) tile(ev.clip).set("physics", "queued");
        else if (i != null) R.add(i, `${base}/${slug}/${ev.file}`, model);
      } else if (ev.type === "budget_stop" && ev.clip) { if (lib) tile(ev.clip).set("fail", short(ev.message || "budget stop", 70)); else if (i != null) R.clip(i, { end: "fail" }); }
      else if (ev.type === "error") status(short(ev.message, 80));
    });
    if (!lib) {
      ids.forEach((c, i) => { if (!R.ended(i)) R.clip(i, { end: "fail" }); });
      await R.gate(Promise.resolve());
      await showPreview(text, R, R.T.runway.some(Boolean) ? "" : "no Runway footage");
    }
    if (!okFoot) { timeStop(); status("", true); paintRoute(rt, dry, "Runway stage failed"); return; }

    const d = await post("/api/data", { slug });
    if (!d.ok) { timeStop(); status("", true); paintRoute(rt, dry, why(d, "Physics stage refused")); return; }
    status("Physics check · 21 gates in MuJoCo");
    const accepted = [];
    let judged = 0;
    await job(d.body.job, (ev) => {
      if (ev.type === "clip_stage" && lib && tiles.has(`${RUN}:${ev.clip}`)) tile(ev.clip).set("physics", ev.stage || "");
      else if (ev.type === "verdict" && ev.clip) {
        judged++;
        if (ev.accepted) accepted.push(ev.clip);
        if (lib) tile(ev.clip).set(ev.accepted ? "ok" : "no", ev.accepted ? `${ev.gates_passed || 21} of ${ev.gates_total || 21} gates` : short(ev.reason || "rejected"));
        else {
          const c = R.T.runway[idx.get(ev.clip)]; if (c) c.verdict = ev.accepted ? "Accepted" : "Rejected";
          R.T.set("physics", "", `Physics · ${accepted.length} of ${judged} accepted`);
        }
      } else if (ev.type === "episodes") status(`${ev.total} episodes for SmolVLA`);
    });
    if (!accepted.length) { timeStop(); status("", true); paintRoute(rt, dry, "No clip passed the physics gates"); return; }

    // SmolVLA on a rented GPU, when the trainer is connected
    const tr = await getJSON("/api/trainer");
    accepted.forEach((c) => tt(c).set("train"));
    if (tr.ok && tr.body.connected) {
      const t = await post("/api/train", { slug, gpu: state.gpu.id });
      if (t.ok) {
        status(`SmolVLA training on ${state.gpu.label}`);
        await job(t.body.job, (ev) => {
          if (ev.type === "train_line") status(`${state.gpu.label} · ${clock(ev.seconds || 0)}${ev.usd != null ? ` · $${Number(ev.usd).toFixed(2)}` : ""}${ev.step && ev.steps ? ` · step ${ev.step}/${ev.steps}` : ""}`);
        });
      }
    }
    timeStop();
    const v = await post("/api/vla", { slug, clips: accepted });
    status(v.ok ? "SmolVLA filming in MuJoCo" : "Waiting for SmolVLA films");
    accepted.forEach((c) => tt(c).set("film"));
    const shown = new Set();
    for (let i = 0; i < 360 && shown.size < accepted.length; i++) {
      const s = await getJSON(`/api/runs/${slug}/vla`);
      if (s.ok) for (const c of accepted) {
        const film = s.body.films[c];
        if (!film || shown.has(c)) continue;
        shown.add(c);
        if (!lib && shown.size > 1) continue; // one tile: the first SmolVLA film replaces the preview
        const t = tt(c); delete t.root.dataset.live; t.root.dataset.vla = "1";
        t.film(`${base}/${slug}/media/vla/${c}.mp4?t=${Date.now()}`, film.success !== false);
      }
      if (shown.size < accepted.length) await wait(5000);
    }
    const mine = [...document.querySelectorAll(`#grid .tile[data-run="${RUN}"]`)];
    const first = mine.find((n) => n.dataset.vla); // the SmolVLA film replaces the Runway footage and the preview
    if (first) mine.forEach((n) => { if (n !== first) n.remove(); });
    if (first && M.seconds != null) tiles.forEach((x) => { if (x.root === first) x.time(M.seconds); });
    status(shown.size ? "" : "Films still rendering", !!shown.size);
    prompts(slug, base);
  }

  // ---------- instant MuJoCo preview: every Train gets a tile with a scripted SO-101 demo of the sentence ----------
  //   generated in this browser (preview.js: MuJoCo WebAssembly + three.js), for any prompt, key or no key. A Runway
  //   clip or SmolVLA film that arrives later replaces it. Scripted motion, never a learned policy.
  const PREVIEW_CAP = "MuJoCo preview · scripted";
  function pvStart(text, instant = true) {
    const rt = state.route || Route.decide(text, { budget: Route.budget });
    state.pvRun = null; // a new Train is always a new run, even when the last one ended at its preview
    startRun(text, "", rt.router);
    state.pvRun = RUN; // the flow's own startRun (replay, cloud, live) then joins this run instead of opening another
    state.pvTile = null;
    if (!instant) return; // a new prompt: runTile() shows Runway progress first; showPreview() mounts it when the gate passes
    const t = tile("preview");
    state.pvTile = t;
    if (!window.RoboHubPreview) { t.set("queued", "preview unavailable"); return; }
    t.set("pv").preview(RoboHubPreview.mount(text));
  }
  // move the preview into the flow's own tile t (unless a real film got there first)
  const attachPreview = (t) => {
    const p = state.pvTile;
    if (!p || p === t || t.real || !p.sim) return;
    const sim = p.sim; p.sim = null;
    p.root.remove(); tiles.delete(p.key);
    t.preview(sim); state.pvTile = t;
  };
  input.addEventListener("focus", () => window.RoboHubPreview && RoboHubPreview.warm(), { once: true });

  // ---------- Train ----------
  $("#composer").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    if (state.running) return;
    const text = textNow();
    state.running = true; $("#go").disabled = true;
    try {
      await ready;
      // library examples replay (hosted) or rerun (local) with the instant preview; anything else loads first
      const lib = state.backend ? await isLibrary(text) : libraryMatch(text);
      pvStart(text, lib);
      await (state.backend ? live(text, lib) : lib ? replay(text) : cloud(text));
    }
    finally { state.running = false; $("#go").disabled = false; }
  });

  // keep every tile film playing while it is on screen (browsers pause muted autoplay offscreen or during load)
  const playIO = new IntersectionObserver((es) => es.forEach((e) => { const v = e.target; if (e.isIntersecting && v.src && v.paused) v.play().catch(() => {}); }), { threshold: 0.2 });
  new MutationObserver(() => document.querySelectorAll("video").forEach((v) => { if (!v.__io) { v.__io = 1; playIO.observe(v); } })).observe(document.body, { childList: true, subtree: true });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) document.querySelectorAll("video").forEach((v) => { if (v.src && v.paused && v.getBoundingClientRect().top < innerHeight) v.play().catch(() => {}); }); });

  // ---------- library: finished runs are already on the page when you scroll down ----------
  ready.then(async () => {
    if (qs.get("preload") === "0" || !state.recorded || state.running) return;
    state.running = true;
    try { await replay(null, true); } finally { state.running = false; }
  });
})();
