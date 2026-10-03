// RoboHub in-browser preview: any sentence -> a small plan -> a scripted SO-101 motion in MuJoCo (WebAssembly),
// drawn live with three.js in the tile. Same motion as src/rohub/preview.py, generated in the visitor's browser: no
// pre-rendered clips, no server, no key.
//
// NOT a learned policy. The arm follows a hand-written script: jaws-down damped-least-squares IK (numeric Jacobian
// through mj_kinematics) tracks eased end-effector waypoints, and the objects are mocap bodies moved kinematically
// (a grasped object rides with the gripper frame, a pushed one slides ahead of the closed jaws, a released one drops
// to its rest height, a book cover turns about its hinge with the held edge). No physics steps run.
//
// The MuJoCo WASM build and three.js come from jsDelivr and load on first use; the SO-101 model (vendored MJCF,
// meshes decimated by scripts/web_meshes.py) is served from sim/so101. Each preview compiles its own model, records
// the motion as per-frame geom poses, frees the model, and loops the recording.
//
//   RoboHubPreview.plan(text) -> plan        RoboHubPreview.summary(plan) -> "place the red block into ..."
//   RoboHubPreview.mount(text) -> {canvas, plan, summary, ready: Promise, stop()}   RoboHubPreview.warm()
(() => {
  "use strict";
  const MUJOCO_URL = "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.13.0/mujoco.js";
  const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js";
  const SIM_DIR = "sim/so101", XML_FILE = "so101_new_calib.xml";
  const FPS = 25;

  // ---------------------------------------------------------------- vocabulary (preview.py)
  const COLORS = {
    red: [0.9, 0.15, 0.12], green: [0.15, 0.7, 0.25], blue: [0.15, 0.35, 0.85], yellow: [0.95, 0.8, 0.1],
    orange: [0.95, 0.5, 0.1], purple: [0.55, 0.25, 0.75], pink: [0.95, 0.45, 0.65], white: [0.93, 0.93, 0.9],
    black: [0.12, 0.12, 0.13], gray: [0.55, 0.55, 0.57], grey: [0.55, 0.55, 0.57], brown: [0.5, 0.32, 0.18],
  };
  const NOUNS = {
    block: "block", cube: "block", brick: "block", blocks: "block", cubes: "block",
    ball: "ball", sphere: "ball", orange: "ball", apple: "ball", marble: "ball",
    cup: "cup", mug: "cup", glass: "cup", bowl: "bowl",
    plate: "plate", dish: "plate", tray: "plate", saucer: "plate",
    square: "pad", pad: "pad", mat: "pad", target: "pad", tape: "pad", zone: "pad", circle: "pad",
    box: "box", bin: "box", basket: "box", container: "box",
    book: "book", lid: "book", laptop: "book", notebook: "book", cover: "book",
  };
  const MOVABLE = new Set(["block", "ball"]);
  const CONTAINERS = new Set(["bowl", "cup", "plate", "pad", "box"]);
  const DEFAULT_COLOR = { block: "red", ball: "orange", cup: "white", bowl: "white", plate: "white", pad: "blue", box: "brown", book: "blue" };
  const SRC_SPOTS = [[0.16, 0.1], [0.15, -0.11], [0.23, 0.13], [0.22, -0.14]];
  const TARGET_SPOT = [0.25, 0.0];
  const BALL_WORDS = new Set(["ball", "sphere", "apple", "marble"]);
  const STOP = new Set(["the", "a", "an", "my", "this", "that", "it", "its", "of", "to", "in", "on", "at", "and", "or", "with", "up", "down", "table", "robot", "arm", "gripper"]);

  const normText = (s) => String(s || "").toLowerCase().replace(/[^a-z0-9 ]+/g, " ").split(/\s+/).filter(Boolean).join(" ");

  function mentions(words) {
    const out = [];
    let col = null;
    for (const w of words) {
      if (w in COLORS) { col = w; continue; }
      if (w in NOUNS) { out.push([NOUNS[w], col, w]); col = null; }
    }
    return out;
  }

  /** Deterministic keyword parser: sentence -> plan (always returns one; `note` says what was approximated). */
  function parse(sentence) {
    const s = normText(sentence);
    let words = s.split(" ").filter(Boolean);
    // "orange" is a colour when a noun follows it, else the fruit (a ball)
    words = words.map((w, i) => (w === "orange" && !(i + 1 < words.length && words[i + 1] in NOUNS) ? "ball" : w));
    const ments = mentions(words);
    let note = "";
    const has = (...ks) => ks.some((k) => new RegExp(`\\b${k}\\b`).test(s));
    const obj = (kind, color) => ({ kind, color: color || DEFAULT_COLOR[kind] });
    const add = (t) => { note = note ? `${note}; ${t}` : t; };

    if (has("open", "close", "shut", "flip open", "lift the lid")) {
      if (ments.some(([k]) => k === "book") || has("lid", "book", "laptop", "cover", "box")) {
        const col = (ments.find(([k, c]) => (k === "book" || k === "box") && c) || [])[1] || null;
        const opening = !has("close", "shut");
        const thing = ["laptop", "box", "lid", "cover"].find((k) => has(k));
        note = has("book", "notebook") || !thing ? "" : `a hinged book stands in for the ${thing}${thing === "box" ? " lid" : ""}`;
        return layout(opening ? "hinge_open" : "hinge_close", [obj("book", col)], [{ op: "hinge", obj: 0, open: opening }], note);
      }
    }

    const blocks = ments.filter(([k]) => MOVABLE.has(k)).map(([, c, w]) => [c, w]);
    const targets = ments.filter(([k]) => CONTAINERS.has(k));

    if (has("tower", "three", "3 blocks", "all the blocks", "all blocks")) {
      const cols = blocks.map(([c]) => c).filter(Boolean);
      for (const c of ["blue", "red", "green", "yellow"]) { if (cols.length >= 3) break; if (!cols.includes(c)) cols.push(c); }
      // bottom first: the last named colour is usually the base ("stack red on green on blue")
      return layout("tower", [obj("block", cols[2]), obj("block", cols[1]), obj("block", cols[0])],
        [{ op: "pick_place", obj: 1, target: 0 }, { op: "pick_place", obj: 2, target: 1 }], note);
    }

    if ((has("stack", "on top") || (blocks.length >= 2 && has("on", "onto"))) && !targets.length) {
      const a = blocks[0] || ["red", "block"];
      const b = blocks[1] || [a[0] !== "blue" ? "blue" : "green", "block"];
      const ka = BALL_WORDS.has(a[1]) ? "ball" : "block";
      return layout("stack", [obj(ka, a[0]), obj("block", b[0])], [{ op: "pick_place", obj: 0, target: 1 }], note);
    }

    let mcol = null, mkind = "block";
    if (blocks.length) {
      mcol = blocks[0][0];
      mkind = NOUNS[blocks[0][1]] === "ball" ? "ball" : "block";
    } else {
      const unknown = [...s.matchAll(/\b(the|a|an|my|this|that)\s+(?:(\w+)\s+)?(\w+)/g)].filter((m) => !(m[3] in NOUNS) && !(m[3] in COLORS) && !STOP.has(m[3]));
      if (unknown.length) {
        mcol = unknown[0][2] in COLORS ? unknown[0][2] : null;
        note = `a block stands in for the ${unknown[0][3]}`;
      }
    }
    let [tkind, tcol] = targets.length ? [targets[0][0], targets[0][1]] : [null, null];

    if (has("push", "slide", "shove", "nudge", "sweep", "drag")) {
      tkind = tkind === "pad" || tkind === "plate" ? tkind : "pad";
      if (targets.length && !["pad", "plate"].includes(targets[0][0])) add(`pushed onto a pad (the ${targets[0][2]} is not pushable into)`);
      return layout("push", [obj(mkind, mcol), obj(tkind, tcol)], [{ op: "push", obj: 0, target: 1 }], note);
    }

    if (has("put", "place", "move", "drop", "set", "transfer", "bring", "carry", "pick", "grab", "take", "load", "insert",
      "throw", "toss", "give", "hand") || targets.length) {
      if (tkind == null) {
        if (has("pick up", "lift", "grab", "take", "raise", "hold") && !has("put", "place", "move")) {
          return layout("lift", [obj(mkind, mcol)], [{ op: "lift", obj: 0 }], note);
        }
        tkind = "pad";
        add("no target named, placed on a pad");
      }
      return layout("pick_place", [obj(mkind, mcol), obj(tkind, tcol)], [{ op: "pick_place", obj: 0, target: 1 }], note);
    }

    if (has("lift", "raise", "hold", "pick")) return layout("lift", [obj(mkind, mcol)], [{ op: "lift", obj: 0 }], note);

    // no object and a gesture verb: the arm waves instead of inventing a block to move
    if (!ments.length && !note && has("wave", "waves", "waving", "hello", "hi", "hey", "greet", "bye", "goodbye", "dance", "salute")) {
      return layout("gesture", [], [{ op: "wave" }], "no object in the sentence, shown as a wave of the gripper");
    }

    add("no known verb, shown as the nearest primitive, pick and place");
    return layout("pick_place", [obj(mkind, mcol), obj("pad", null)], [{ op: "pick_place", obj: 0, target: 1 }], note, true);
  }

  /** Name the objects and place them on the table (movables at the source spots, the target at the far spot). */
  function layout(family, objs, steps, note = "", fallback = false) {
    let src = 0;
    objs.forEach((o, i) => { o.name = `o${i}`; });
    if (family === "tower") {
      objs[0].xy = [0.21, 0.0];
      objs[1].xy = [...SRC_SPOTS[0]];
      objs[2].xy = [...SRC_SPOTS[1]];
    } else if (family.startsWith("hinge")) {
      objs[0].xy = [0.21, 0.0];
    } else {
      objs.forEach((o, i) => { o.xy = steps.some((st) => st.target === i) ? [...TARGET_SPOT] : [...SRC_SPOTS[src++]]; });
    }
    for (const st of steps) for (const k of ["obj", "target"]) if (typeof st[k] === "number") st[k] = `o${st[k]}`;
    const plan = { family, objects: objs, steps, note };
    if (fallback) plan.fallback = true;
    return plan;
  }

  function summary(plan) {
    const by = Object.fromEntries(plan.objects.map((o) => [o.name, o]));
    const name = (o) => `${o.color} ${o.kind}`;
    return plan.steps.map((st) => {
      if (st.op === "wave") return "wave the gripper";
      const o = by[st.obj];
      if (st.op === "hinge") return `${st.open !== false ? "open" : "close"} the ${name(o)}`;
      if (st.op === "lift") return `lift the ${name(o)}`;
      const t = by[st.target];
      const prep = ["bowl", "cup", "box"].includes(t.kind) ? "into" : "onto";
      return `${st.op === "push" ? "push" : "place"} the ${name(o)} ${prep} the ${name(t)}`;
    }).join(", then ");
  }

  // ---------------------------------------------------------------- scene (preview.build + scene.build_spec)
  const HALF = 0.015; // block half size / ball radius
  const SIZES = { block: 2 * HALF, ball: 2 * HALF, pad: 0.001, plate: 0.006, bowl: 0.004, cup: 0.004, box: 0.004 };
  const RIM = { bowl: 0.028, cup: 0.042, box: 0.035 };
  const BOOK_L = 0.06, BOOK_W = 0.03, BOOK_T = 0.008;
  const COVER_OPEN = 1.75;

  const f5 = (a) => a.map((v) => +Number(v).toFixed(5)).join(" ");
  const rgbaOf = (c) => f5([...(COLORS[c] || [0.6, 0.6, 0.6]), 1]);
  const VIS = 'contype="0" conaffinity="0"';
  const geom = (type, size, pos, rgba, extra = "") => `<geom type="${type}" size="${f5(size)}" pos="${f5(pos)}" rgba="${rgba}" ${VIS} ${extra}/>`;

  function objectXml(o) {
    const [x, y] = o.xy, c = rgbaOf(o.color), n = o.name;
    const body = (z, inner, name = n) => `<body name="${name}" pos="${f5([x, y, z])}" mocap="true">${inner}</body>`;
    switch (o.kind) {
      case "block": return body(HALF, geom("box", [HALF, HALF, HALF], [0, 0, 0], c));
      case "ball": return body(HALF, geom("sphere", [HALF], [0, 0, 0], c));
      case "pad": {
        const t = 0.04, w = 0.006;
        return body(0, [[t, 0, w, t + w], [-t, 0, w, t + w], [0, t, t + w, w], [0, -t, t + w, w]]
          .map(([px, py, sx, sy]) => geom("box", [sx, sy, 0.0006], [px, py, 0.0006], c)).join(""));
      }
      case "plate": return body(0, geom("cylinder", [0.06, 0.003], [0, 0, 0.003], c) + geom("cylinder", [0.066, 0.0015], [0, 0, 0.0045], "0.8 0.8 0.78 1"));
      case "bowl": case "cup": {
        const [r, h, t] = o.kind === "bowl" ? [0.05, RIM.bowl, 0.006] : [0.033, RIM.cup, 0.005];
        const segs = 20, half = (r + t / 2) * Math.tan(Math.PI / segs) * 1.08;
        let g = geom("cylinder", [r + t, 0.002], [0, 0, 0.002], c);
        for (let i = 0; i < segs; i++) {
          const a = (2 * Math.PI * i) / segs;
          g += geom("box", [t / 2, half, h / 2], [(r + t / 2) * Math.cos(a), (r + t / 2) * Math.sin(a), h / 2], c, `quat="${f5([Math.cos(a / 2), 0, 0, Math.sin(a / 2)])}"`);
        }
        if (o.kind === "cup") g += geom("box", [0.004, 0.012, 0.014], [0, -(r + 0.016), h * 0.55], c);
        return body(0, g);
      }
      case "box": {
        const hb = 0.05, h = RIM.box, t = 0.005;
        return body(0, geom("box", [hb, hb, 0.002], [0, 0, 0.002], c) + [[hb, 0, t, hb], [-hb, 0, t, hb], [0, hb, hb, t], [0, -hb, hb, t]]
          .map(([px, py, sx, sy]) => geom("box", [sx, sy, h / 2], [px, py, h / 2], c)).join(""));
      }
      case "book": {
        // pages (fixed) + a cover (mocap) hinged along the book's -y edge; the tab sits on the +y edge
        const L = BOOK_L, W = BOOK_W, T = BOOK_T;
        const base = body(0, geom("box", [L, W, T], [0, 0, T], "0.95 0.93 0.85 1") + geom("box", [L, W, 0.0015], [0, 0, 0.0015], c));
        const cover = `<body name="${n}_cover" pos="${f5([x, y - W, 2 * T + 0.002])}" mocap="true">${geom("box", [L, W, 0.002], [0, W, 0], c)}${geom("box", [0.012, 0.006, 0.004], [0, 2 * W + 0.004, 0], "0.95 0.8 0.2 1")}</body>`;
        return base + cover;
      }
      default: return "";
    }
  }

  function sceneXml(baseXml, plan) {
    const world = [
      `<geom name="table" type="box" size="0.45 0.6 0.02" pos="0.2 0 -0.02" rgba="0.62 0.52 0.40 1" ${VIS}/>`,
      ...plan.objects.map(objectXml),
    ].join("\n");
    return baseXml
      .replace(/meshdir="[^"]*"/, 'meshdir="so101/assets"')
      .replace(/\s*<geom [^>]*class="collision"[^>]*\/>/g, "") // the preview draws; nothing collides
      .replace("<worldbody>", `<worldbody>\n${world}`);
  }

  // ---------------------------------------------------------------- script (preview.Animator)
  const JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"];
  const HOME = [0.0, -0.6, 0.9, 0.9, 0.0];
  const GRIPPER_CLOSED = -0.1745, GRIP_OPEN = 0.7;
  const CENTRE_OPEN = 0.024, CENTRE_GRASP = 0.0135, GRIP_DEPTH = 0.012;
  const W_ORI = 0.1, DAMP = 1e-4, W_POST = 1e-5;

  const ease = (s) => s * s * (3 - 2 * s);
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const siteFor = (c, h, centre) => [c[0] - centre * h[0], c[1] - centre * h[1], c[2] - GRIP_DEPTH - centre * h[2]];
  // row-major 3x3 helpers (MuJoCo xmat layout)
  const mulTv = (R, v) => [0, 1, 2].map((i) => R[i] * v[0] + R[3 + i] * v[1] + R[6 + i] * v[2]); // R^T v
  const mulv = (R, v) => [0, 1, 2].map((i) => R[3 * i] * v[0] + R[3 * i + 1] * v[1] + R[3 * i + 2] * v[2]);
  const mulTm = (A, B) => { const o = new Array(9); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) o[3 * i + j] = A[i] * B[j] + A[3 + i] * B[3 + j] + A[6 + i] * B[6 + j]; return o; };
  const mulm = (A, B) => { const o = new Array(9); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) o[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] + A[3 * i + 2] * B[6 + j]; return o; };
  function quat2mat([w, x, y, z]) {
    return [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)];
  }
  function mat2quat(m) {
    const tr = m[0] + m[4] + m[8];
    let q;
    if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; q = [0.25 * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s]; }
    else if (m[0] > m[4] && m[0] > m[8]) { const s = Math.sqrt(1 + m[0] - m[4] - m[8]) * 2; q = [(m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s]; }
    else if (m[4] > m[8]) { const s = Math.sqrt(1 + m[4] - m[0] - m[8]) * 2; q = [(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s]; }
    else { const s = Math.sqrt(1 + m[8] - m[0] - m[4]) * 2; q = [(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s]; }
    const n = Math.hypot(...q);
    return q.map((v) => v / n);
  }
  /** A x = b by Gaussian elimination with partial pivoting (A is 5 x 5). */
  function solveLinear(A, b) {
    const n = b.length, M = A.map((row, i) => [...row, b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let k = c + 1; k < n; k++) if (Math.abs(M[k][c]) > Math.abs(M[p][c])) p = k;
      [M[c], M[p]] = [M[p], M[c]];
      for (let k = c + 1; k < n; k++) { const f = M[k][c] / M[c][c]; for (let j = c; j <= n; j++) M[k][j] -= f * M[c][j]; }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; }
    return x;
  }

  class Script {
    constructor(mj, model, plan) {
      this.mj = mj; this.m = model; this.plan = plan;
      this.d = new mj.MjData(model);
      this.s = new mj.MjData(model); // scratch for the IK
      const id = (type, name) => {
        const i = mj.mj_name2id(model, mj.mjtObj[type].value, name);
        if (i < 0) throw new Error(`preview: no ${name} in the model`);
        return i;
      };
      this.site = id("mjOBJ_SITE", "gripperframe");
      const jids = JOINTS.map((j) => id("mjOBJ_JOINT", j));
      this.qadr = jids.map((j) => model.jnt_qposadr[j]);
      this.lo = jids.map((j) => model.jnt_range[2 * j]);
      this.hi = jids.map((j) => model.jnt_range[2 * j + 1]);
      this.gadr = model.jnt_qposadr[id("mjOBJ_JOINT", "gripper")];
      this.mid = {};
      this.kind = {};
      for (const o of plan.objects) {
        this.kind[o.name] = o.kind;
        this.mid[o.name] = model.body_mocapid[id("mjOBJ_BODY", o.name)];
        if (o.kind === "book") this.mid[`${o.name}_cover`] = model.body_mocapid[id("mjOBJ_BODY", `${o.name}_cover`)];
      }
      this.ngeom = model.ngeom;
      this.heading = [1, 0, 0];
      this.q = HOME.slice();
      this.g = GRIPPER_CLOSED;
      this.homeEE = this.fk(HOME).slice(0, 3);
      this.ee = this.homeEE.slice();
      this.held = null; this.falling = new Map(); this.push = null; this.hinge = null; this.coverFall = null;
      this.maxErr = 0;
      this.frames = [];
    }

    /** [site pos, site x axis (jaw direction), site z axis (closing axis)] at arm joints q, on the scratch data. */
    fk(q) {
      const s = this.s, qp = s.qpos;
      for (let i = 0; i < 5; i++) qp[this.qadr[i]] = q[i];
      this.mj.mj_kinematics(this.m, s);
      const p = s.site_xpos, R = s.site_xmat, o = this.site * 3, r = this.site * 9;
      return [p[o], p[o + 1], p[o + 2], R[r], R[r + 3], R[r + 6], R[r + 2], R[r + 5], R[r + 8]];
    }

    /** Damped least squares: site position, jaws straight down, closing axis on the horizontal heading (GraspIK). */
    solve(target, q0, iters = 40) {
      let q = q0.slice();
      const h = this.heading, want = [target[0], target[1], target[2], 0, 0, -1, h[0], h[1], h[2]];
      const wt = [1, 1, 1, W_ORI, W_ORI, W_ORI, W_ORI, W_ORI, W_ORI];
      for (let it = 0; it < iters; it++) {
        const f = this.fk(q);
        const e = want.map((v, i) => (v - f[i]) * wt[i]);
        const ep = Math.hypot(e[0], e[1], e[2]);
        if (ep < 5e-4 && Math.hypot(...e.slice(3)) / W_ORI < 0.02) break;
        const J = [];
        for (let j = 0; j < 5; j++) {
          const qq = q.slice(); qq[j] += 1e-5;
          const f2 = this.fk(qq);
          J.push(f2.map((v, i) => ((v - f[i]) / 1e-5) * wt[i]));
        }
        const A = [], b = [];
        for (let a = 0; a < 5; a++) {
          A.push([]);
          for (let c = 0; c < 5; c++) { let sum = 0; for (let i = 0; i < 9; i++) sum += J[a][i] * J[c][i]; A[a].push(sum + (a === c ? DAMP + W_POST : 0)); }
          let sb = 0; for (let i = 0; i < 9; i++) sb += J[a][i] * e[i];
          b.push(sb + W_POST * (HOME[a] - q[a]));
        }
        const dq = solveLinear(A, b);
        q = q.map((v, i) => clamp(v + clamp(dq[i], -0.2, 0.2), this.lo[i], this.hi[i]));
        if (Math.hypot(...dq) < 1e-6) break;
      }
      const f = this.fk(q);
      return [q, Math.hypot(target[0] - f[0], target[1] - f[1], target[2] - f[2])];
    }

    pos(n) { const p = this.d.mocap_pos, k = this.mid[n] * 3; return [p[k], p[k + 1], p[k + 2]]; }
    sitePose() {
      const d = this.d, o = this.site * 3, r = this.site * 9;
      return [[d.site_xpos[o], d.site_xpos[o + 1], d.site_xpos[o + 2]], Array.from(d.site_xmat.subarray(r, r + 9))];
    }
    setQuat(mid, q) { const mq = this.d.mocap_quat; for (let i = 0; i < 4; i++) mq[mid * 4 + i] = q[i]; }

    apply() {
      const { mj, m, d } = this;
      const qp = d.qpos;
      for (let i = 0; i < 5; i++) qp[this.qadr[i]] = this.q[i];
      qp[this.gadr] = this.g;
      mj.mj_kinematics(m, d);
      const [sp, sr] = this.sitePose();
      const mp = d.mocap_pos;
      if (this.held) {
        const [n, relP, relR] = this.held, k = this.mid[n];
        const off = mulv(sr, relP);
        for (let i = 0; i < 3; i++) mp[k * 3 + i] = sp[i] + off[i];
        this.setQuat(k, mat2quat(mulm(sr, relR)));
      }
      for (const [n, st] of this.falling) {
        const k = this.mid[n] * 3 + 2;
        st[1] += 9.8 / FPS;
        mp[k] = Math.max(st[0], mp[k] - st[1] / FPS);
        if (mp[k] <= st[0]) this.falling.delete(n);
      }
      if (this.push) {
        const [n, o0, dir, reach, goal] = this.push, k = this.mid[n] * 3;
        const s = (sp[0] - o0[0]) * dir[0] + (sp[1] - o0[1]) * dir[1] + reach;
        const off = Math.min(Math.max(0, s), goal);
        mp[k] = o0[0] + dir[0] * off; mp[k + 1] = o0[1] + dir[1] * off;
      }
      if (this.coverFall) {
        let [cid, ang, vel, goal] = this.coverFall;
        vel += 12 / FPS;
        ang = goal > ang ? Math.min(goal, ang + vel / FPS) : Math.max(goal, ang - vel / FPS);
        this.coverFall = ang === goal ? null : [cid, ang, vel, goal];
        this.hinge = [cid, ang];
      }
      if (this.hinge) { const [cid, a] = this.hinge; this.setQuat(cid, [Math.cos(a / 2), Math.sin(a / 2), 0, 0]); } // about +x
      mj.mj_kinematics(m, d);
    }

    frame() {
      this.apply();
      const d = this.d;
      this.frames.push([Float32Array.from(d.geom_xpos), Float32Array.from(d.geom_xmat)]);
    }

    *move(goal, dur = 0.6, grip = null, path = null) {
      const n = Math.max(1, Math.round(dur * FPS)), e0 = this.ee.slice(), g0 = this.g;
      for (let i = 1; i <= n; i++) {
        const s = ease(i / n);
        const ee = path ? path(s) : e0.map((v, k) => v + (goal[k] - v) * s);
        ee[2] = Math.min(ee[2], 0.09); // the jaws-down reach ceiling
        this.ee = ee;
        if (grip != null) this.g = g0 + (grip - g0) * s;
        const [q, err] = this.solve(ee, this.q);
        this.q = q; this.maxErr = Math.max(this.maxErr, err);
        this.frame(); yield;
      }
    }
    *hold(dur) { for (let i = 0; i < Math.round(dur * FPS); i++) { this.frame(); yield; } }
    here() { return this.ee.slice(); }
    up(dz) { return [this.ee[0], this.ee[1], this.ee[2] + dz]; }

    *grasp(n) {
      const c = this.pos(n), r = Math.hypot(c[0], c[1]);
      this.heading = [c[0] / r, c[1] / r, 0];
      const above = siteFor(c, this.heading, CENTRE_OPEN);
      yield* this.move([above[0], above[1], 0.075], 0.8, GRIP_OPEN);
      yield* this.move(above, 0.45);
      yield* this.move(siteFor(c, this.heading, CENTRE_GRASP), 0.15);
      yield* this.move(this.here(), 0.3, GRIPPER_CLOSED + 0.35);
      const [sp, sr] = this.sitePose();
      const k = this.mid[n], mq = this.d.mocap_quat;
      const orr = quat2mat([mq[k * 4], mq[k * 4 + 1], mq[k * 4 + 2], mq[k * 4 + 3]]);
      this.held = [n, mulTv(sr, [c[0] - sp[0], c[1] - sp[1], c[2] - sp[2]]), mulTm(sr, orr)];
    }
    *place(n, target) {
      const t = this.pos(target), tk = this.kind[target];
      const top = t[2] + (tk === "block" || tk === "ball" ? HALF : SIZES[tk]);
      const rest = top + HALF;
      const release = Math.max(rest + 0.006, (RIM[tk] || 0) + HALF + 0.008);
      const carry = Math.min(Math.max(0.075, release + 0.015), 0.098);
      const c = this.pos(n), off = this.ee.map((v, i) => v - c[i]);
      yield* this.move([c[0] + off[0], c[1] + off[1], carry + off[2]], 0.45);
      yield* this.move([t[0] + off[0], t[1] + off[1], carry + off[2]], 0.8);
      yield* this.move([t[0] + off[0], t[1] + off[1], release + off[2]], 0.4);
      this.held = null;
      this.falling.set(n, [rest, 0]);
      yield* this.move(this.here(), 0.25, GRIP_OPEN);
      yield* this.move(this.up(0.05), 0.35);
    }
    *lift(n) {
      yield* this.grasp(n);
      const c = this.pos(n), off = this.ee.map((v, i) => v - c[i]);
      yield* this.move([c[0] + off[0], c[1] + off[1], 0.095 + off[2]], 0.7);
      yield* this.hold(0.4);
      yield* this.move([c[0] + off[0], c[1] + off[1], HALF + 0.004 + off[2]], 0.6);
      this.held = null;
      this.falling.set(n, [HALF, 0]);
      yield* this.move(this.here(), 0.25, GRIP_OPEN);
      yield* this.move(this.up(0.05), 0.35);
    }
    *pushTo(n, target) {
      const o = this.pos(n), t = this.pos(target);
      const dist = Math.hypot(t[0] - o[0], t[1] - o[1]), dir = [(t[0] - o[0]) / dist, (t[1] - o[1]) / dist];
      const back = HALF + 0.012, r = Math.hypot(o[0], o[1]), z = 0.012;
      this.heading = [o[0] / r, o[1] / r, 0]; // the natural wrist roll; closed jaws push
      const start = [o[0] - dir[0] * (back + 0.03), o[1] - dir[1] * (back + 0.03), z];
      const end = [t[0] - dir[0] * back, t[1] - dir[1] * back, z];
      yield* this.move([start[0], start[1], 0.07], 0.8, GRIPPER_CLOSED);
      yield* this.move(start, 0.4);
      this.push = [n, o, dir, back, dist];
      yield* this.move(end, 1.3);
      this.push = null;
      yield* this.move([end[0], end[1], end[2] + 0.06], 0.4);
    }
    *hingeBook(n, opening) {
      const cid = this.mid[`${n}_cover`], mp = this.d.mocap_pos;
      const hp = [mp[cid * 3], mp[cid * 3 + 1], mp[cid * 3 + 2]]; // hinge line point (x centre, -y edge, cover height)
      const R = 2 * BOOK_W + 0.004; // hinge -> tab
      // opening: lift the free edge past vertical, let go, it falls open flat; closing: the mirror image
      const [a0, a1] = opening ? [0, COVER_OPEN] : [Math.PI, Math.PI - COVER_OPEN];
      this.hinge = [cid, a0];
      this.apply();
      this.heading = [1, 0, 0]; // jaws close across the tab along x
      const grip = (a) => [hp[0], hp[1] + R * Math.cos(a), hp[2] + R * Math.sin(a) - 0.004];
      yield* this.move([...grip(a0).slice(0, 2), grip(a0)[2] + 0.05], 0.8, GRIP_OPEN);
      yield* this.move(grip(a0), 0.4);
      yield* this.move(this.here(), 0.25, GRIPPER_CLOSED + 0.4);
      yield* this.move(null, 1.6, null, (s) => { const a = a0 + (a1 - a0) * s; this.hinge = [cid, a]; return grip(a); });
      this.coverFall = [cid, a1, 0, opening ? Math.PI : 0];
      yield* this.move(this.here(), 0.25, GRIP_OPEN);
      yield* this.move(this.up(0.04), 0.3);
    }
    *wave() {
      this.heading = [1, 0, 0];
      yield* this.move([0.2, 0.0, 0.085], 0.7, GRIP_OPEN);
      for (let i = 0; i < 2; i++) { yield* this.move([0.19, 0.1, 0.085], 0.5); yield* this.move([0.19, -0.1, 0.085], 0.6); }
      yield* this.move([0.2, 0.0, 0.085], 0.4);
      for (let i = 0; i < 2; i++) { yield* this.move(this.here(), 0.2, GRIPPER_CLOSED); yield* this.move(this.here(), 0.2, GRIP_OPEN); }
    }

    *run() {
      [this.q] = this.solve(this.homeEE, HOME, 80);
      if (this.plan.family === "hinge_close") this.hinge = [this.mid[`${this.plan.objects[0].name}_cover`], Math.PI];
      yield* this.hold(0.3);
      for (const st of this.plan.steps) {
        if (st.op === "pick_place") { yield* this.grasp(st.obj); yield* this.place(st.obj, st.target); }
        else if (st.op === "push") yield* this.pushTo(st.obj, st.target);
        else if (st.op === "lift") yield* this.lift(st.obj);
        else if (st.op === "hinge") yield* this.hingeBook(st.obj, st.open !== false);
        else if (st.op === "wave") yield* this.wave();
      }
      yield* this.move(this.homeEE, 0.8, GRIPPER_CLOSED);
      yield* this.hold(0.5);
    }

    free() { this.d.delete(); this.s.delete(); this.m.delete(); }
  }

  // ---------------------------------------------------------------- runtime (lazy: first warm() or mount())
  let runtime = null;
  function load() {
    runtime ||= (async () => {
      const [mjMod, THREE, xml] = await Promise.all([
        import(MUJOCO_URL).then((m) => m.default()),
        import(THREE_URL),
        fetch(`${SIM_DIR}/${XML_FILE}`).then((r) => { if (!r.ok) throw new Error("SO-101 model missing"); return r.text(); }),
      ]);
      const files = [...xml.matchAll(/<mesh [^>]*file="([^"]+)"/g)].map((m) => m[1]);
      const fs = mjMod.FS;
      for (const d of ["/so101", "/so101/assets"]) { try { fs.mkdir(d); } catch { /* already there */ } }
      await Promise.all(files.map(async (f) => {
        const r = await fetch(`${SIM_DIR}/assets/${f}`);
        if (!r.ok) throw new Error(`mesh ${f} missing`);
        fs.writeFile(`/so101/assets/${f}`, new Uint8Array(await r.arrayBuffer()));
      }));
      // one WebGL context for every tile: each preview renders into it and copies the picture to its own 2D canvas
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "low-power" });
      renderer.setPixelRatio(1);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      return { mj: mjMod, THREE, xml, renderer, geoms: new Map() };
    })();
    runtime.catch(() => { runtime = null; });
    return runtime;
  }

  const MESH = 7, BOX = 6, CYLINDER = 5, SPHERE = 2;
  function buildScene(rt, model) {
    const { THREE } = rt;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color().setRGB(0.055, 0.063, 0.078, THREE.SRGBColorSpace);
    scene.add(new THREE.HemisphereLight(0xdfe6ff, 0x2a2622, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(0.5, -0.35, 1.3);
    key.target.position.set(0.18, 0, 0);
    key.castShadow = true;
    Object.assign(key.shadow.camera, { left: -0.4, right: 0.4, top: 0.4, bottom: -0.4, near: 0.2, far: 3 });
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.bias = -0.0005;
    scene.add(key, key.target);
    const fill = new THREE.DirectionalLight(0xffffff, 0.6);
    fill.position.set(-0.3, 0.5, 1.2);
    scene.add(fill);
    const meshGeo = (id) => {
      let g = rt.geoms.get(id); // arm meshes are identical in every compiled scene
      if (!g) {
        const v0 = model.mesh_vertadr[id], f0 = model.mesh_faceadr[id];
        const idx = new THREE.BufferGeometry();
        idx.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(model.mesh_vert.subarray(v0 * 3, (v0 + model.mesh_vertnum[id]) * 3)), 3));
        idx.setIndex(new THREE.BufferAttribute(Uint32Array.from(model.mesh_face.subarray(f0 * 3, (f0 + model.mesh_facenum[id]) * 3)), 1));
        g = idx.toNonIndexed();
        g.computeVertexNormals();
        idx.dispose();
        rt.geoms.set(id, g);
      }
      return g;
    };
    const own = [];
    const meshes = [];
    for (let i = 0; i < model.ngeom; i++) {
      const type = model.geom_type[i], s = [0, 1, 2].map((k) => model.geom_size[i * 3 + k]);
      let geo;
      if (type === MESH) geo = meshGeo(model.geom_dataid[i]);
      else {
        if (type === BOX) geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
        else if (type === CYLINDER) geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 40).rotateX(Math.PI / 2);
        else if (type === SPHERE) geo = new THREE.SphereGeometry(s[0], 28, 18);
        else geo = new THREE.BufferGeometry();
        own.push(geo);
      }
      const mat = model.geom_matid[i];
      const c = [0, 1, 2].map((k) => (mat >= 0 ? model.mat_rgba[mat * 4 + k] : model.geom_rgba[i * 4 + k]));
      const material = new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace), roughness: 0.55, metalness: type === MESH ? 0.1 : 0.0, flatShading: type === MESH });
      const mesh = new THREE.Mesh(geo, material);
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      scene.add(mesh);
      meshes.push(mesh);
    }
    // the hero camera of the mp4 previews (media.hero_cam): lookat (0.19, -0.035, 0.09), 0.58 m, azimuth -155, elevation -24
    const camera = new THREE.PerspectiveCamera(45, 16 / 10, 0.02, 10);
    const az = (-155 * Math.PI) / 180, el = (-24 * Math.PI) / 180, look = [0.19, -0.035, 0.09], dist = 0.62;
    camera.up.set(0, 0, 1);
    camera.position.set(look[0] - dist * Math.cos(el) * Math.cos(az), look[1] - dist * Math.cos(el) * Math.sin(az), look[2] - dist * Math.sin(el));
    camera.lookAt(...look);
    const dispose = () => { own.forEach((g) => g.dispose()); meshes.forEach((m) => m.material.dispose()); };
    return { scene, camera, meshes, dispose };
  }

  // ---------------------------------------------------------------- previews + one shared animation loop
  const active = new Set();
  let raf = 0;
  const kick = () => { if (!raf && active.size) raf = requestAnimationFrame(loop); };
  function loop(now) {
    raf = 0;
    for (const p of active) p.tick(now);
    kick();
  }
  const io = "IntersectionObserver" in window ? new IntersectionObserver((es) => es.forEach((e) => { if (e.target.__pv) e.target.__pv.visible = e.isIntersecting; }), { rootMargin: "100px" }) : null;

  class Preview {
    constructor(text) {
      this.plan = parse(text);
      this.summary = summary(this.plan);
      this.note = this.plan.note || "";
      this.canvas = document.createElement("canvas");
      this.canvas.className = "tile-sim";
      this.canvas.style.cssText = "display:block;width:100%;height:100%"; // sized by its box, never by its own pixels
      this.canvas.setAttribute("role", "img");
      this.canvas.setAttribute("aria-label", `Scripted MuJoCo preview: ${this.summary}`);
      this.canvas.__pv = this;
      this.visible = !io;
      if (io) io.observe(this.canvas);
      this.ctx = this.canvas.getContext("2d");
      this.stopped = false;
      this.seen = false;
      this.t0 = performance.now();
      this.ready = this.start();
    }
    async start() {
      const rt = await load();
      if (this.stopped) return null;
      const model = rt.mj.MjModel.from_xml_string(sceneXml(rt.xml, this.plan));
      this.view = buildScene(rt, model);
      this.rt = rt;
      this.script = new Script(rt.mj, model, this.plan);
      this.gen = this.script.run();
      this.gen.next(); // first frame
      this.frames = this.script.frames;
      return new Promise((resolve) => { this.onFirst = resolve; active.add(this); kick(); });
    }
    /** Record more of the motion (cheap: tens of IK solves per animation frame), then free MuJoCo when it is done. */
    advance(upTo) {
      if (!this.gen) return;
      const t = performance.now();
      while (this.frames.length <= upTo && performance.now() - t < 8) {
        if (this.gen.next().done) { this.done(); return; }
      }
    }
    done() {
      this.gen = null;
      this.maxErr = this.script.maxErr;
      this.script.free();
      this.script = null;
    }
    tick(now) {
      if (this.stopped) return;
      if (!this.canvas.isConnected) { if (this.seen) this.stop(); return; }
      this.seen = true;
      if (this.playStart == null) this.playStart = now;
      let i = Math.floor(((now - this.playStart) / 1000) * FPS);
      if (this.gen) { this.advance(i + 2); if (this.gen) i = Math.min(i, this.frames.length - 1); }
      if (!this.gen) i %= this.frames.length;
      if (!this.visible && this.drawn) return;
      this.draw(this.frames[i]);
    }
    draw([xpos, xmat]) {
      const { renderer } = this.rt, { scene, camera, meshes } = this.view, c = this.canvas;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = clamp(Math.round(c.clientWidth * dpr), 2, 1600), h = clamp(Math.round(c.clientHeight * dpr), 2, 1000);
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
      const size = renderer.getSize(new this.rt.THREE.Vector2());
      if (size.x !== w || size.y !== h) renderer.setSize(w, h, false);
      if (Math.abs(camera.aspect - w / h) > 1e-3) { camera.aspect = w / h; camera.updateProjectionMatrix(); }
      meshes.forEach((m, g) => {
        const o = g * 9, p = g * 3;
        m.matrix.set(xmat[o], xmat[o + 1], xmat[o + 2], xpos[p], xmat[o + 3], xmat[o + 4], xmat[o + 5], xpos[p + 1], xmat[o + 6], xmat[o + 7], xmat[o + 8], xpos[p + 2], 0, 0, 0, 1);
        m.matrixWorldNeedsUpdate = true;
      });
      renderer.render(scene, camera);
      this.ctx.drawImage(renderer.domElement, 0, 0, w, h);
      if (!this.drawn) {
        this.drawn = true;
        this.firstFrameMs = Math.round(performance.now() - this.t0);
        c.dataset.firstFrameMs = String(this.firstFrameMs);
        if (this.onFirst) { this.onFirst(this); this.onFirst = null; }
      }
    }
    stop() {
      if (this.stopped) return;
      this.stopped = true;
      active.delete(this);
      if (this.onFirst) { this.onFirst(null); this.onFirst = null; } // stopped before its first frame: settle `ready`
      if (io) io.unobserve(this.canvas);
      if (this.script) { this.script.free(); this.script = null; }
      this.gen = null;
      this.frames = [];
      if (this.view) { this.view.dispose(); this.view = null; }
      this.canvas.remove();
    }
  }

  window.RoboHubPreview = {
    plan: parse,
    summary,
    warm() { load().catch(() => {}); },
    mount(text) { return new Preview(text); },
  };
})();
