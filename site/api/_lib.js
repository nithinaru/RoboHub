// Shared helpers for the live Runway path (Vercel functions). Bring your own key: every call uses the visitor's
// Runway API key from the X-Runway-Key header. It is forwarded to Runway only, never stored or logged.
const BASE = "https://api.dev.runwayml.com/v1";
const VERSION = "2024-11-06";
const PER_IP = 12; // calls per visitor per 10 minutes (per warm instance)
const hits = new Map();

const fail = (msg, code) => Object.assign(new Error(msg), { code });

function keyOf(req) {
  const k = String(req.headers["x-runway-key"] || "").trim();
  if (!k) throw fail("Add your Runway API key to generate new prompts.", 401);
  if (!/^key_[A-Za-z0-9_-]{16,300}$/.test(k)) throw fail("That doesn't look like a Runway API key (it starts with key_).", 401);
  return k;
}

async function runway(key, method, path, body) {
  const r = await fetch(BASE + path, {
    method,
    headers: { Authorization: `Bearer ${key}`, "X-Runway-Version": VERSION, "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const j = await r.json().catch(() => ({}));
  if (r.status === 401 || r.status === 403) throw fail("Runway rejected this API key.", 401);
  if (!r.ok) throw fail(j.error || j.message || `Runway HTTP ${r.status}`, r.status === 400 || r.status === 404 ? r.status : 502);
  return j;
}

// budget routing (same as route.js): a run is 5 clips; every budget is a quality-optimized Model Router whose price
// ceiling is (budget / 5 clips) credits a clip, created on the visitor's account the first time it is used
const BUDGETS = [2, 4, 6, 8, 10], CLIPS = 5;
function decide(_task, budget) {
  const usd = BUDGETS.includes(Number(budget)) ? Number(budget) : 2;
  const ceiling = Math.round((usd * 100) / CLIPS);
  return { router: `robohub-q${ceiling}`, budget: usd, ceiling, reasons: [`best quality up to ${ceiling} credits a clip`] };
}
async function ensureRouter(key, router, ceiling) {
  const have = await runway(key, "GET", "/routers");
  if ((have.data || []).some((r) => r.slug === router)) return router;
  const made = await runway(key, "POST", "/routers", {
    slug: router,
    name: router,
    description: `RoboHub demonstrations: best quality at up to ${ceiling} credits a clip`,
    settings: { schemaVersion: 1, models: { mode: "allow_new_except", ids: [] }, optimizeFor: "quality", maxCreditsPerGeneration: { video: ceiling }, fallback: { onCapacity: true } },
  });
  return made.slug || router;
}

function clean(prompt) {
  const p = String(prompt || "").replace(/\s+/g, " ").trim().replace(/[.!]+$/, "");
  if (p.length < 6 || p.length > 140 || !/^[a-zA-Z0-9 ,'-]+$/.test(p)) {
    throw fail("Use a short plain sentence (6 to 140 letters), like \"put the green cup on the plate\".", 400);
  }
  return p;
}

// PhyT2V-style positive phrasing, generalized from plan.py's v1 prompts. A run makes 3 demonstration clips; each
// variant restyles the scene (table, light, camera angle, like plan.py's variation axes) so the clips differ.
const SCENES = [
  { table: "a plain light oak wooden tabletop", light: "Soft natural daylight from a window on the left", angle: "three-quarter view from slightly above" },
  { table: "a matte white laminate desk", light: "Warm late-afternoon sunlight with long soft shadows", angle: "front view from slightly to the left, about 35 degrees down" },
  { table: "a dark walnut table", light: "Bright even studio light", angle: "low front view from table height, about 25 degrees down" },
];
function variantOf(v) {
  if (v == null || v === "") return 0;
  const n = Number(v);
  if (!Number.isInteger(n) || n < 0 || n >= SCENES.length) throw fail(`variant must be 0 to ${SCENES.length - 1}`, 400);
  return n;
}
const imagePrompt = (task, variant = 0) => {
  const s = SCENES[variant] || SCENES[0];
  return `Photorealistic photo, ${s.angle}, of ${s.table} holding only the few simple objects needed to ${task}. ` +
    `Nothing else is on the table. A person's right hand hovers just above the first object, fingers open, ready to ` +
    `grasp it; the forearm enters from the right edge of the frame. The whole hand and every object are fully in ` +
    `frame and unobstructed. ${s.light}. Sharp focus, no text.`;
};
const videoPrompt = (task) =>
  `The right hand does this: ${task}. One continuous smooth motion at natural speed, then the open hand moves back up ` +
  `and away. Static camera, locked off. Every object keeps its shape, size and count the whole time.`;

function limit(req) {
  const ip = String(req.headers["x-forwarded-for"] || "").split(",")[0].trim() || "?";
  const now = Date.now(), recent = (hits.get(ip) || []).filter((t) => now - t < 600000);
  if (recent.length >= PER_IP) throw fail("That's a lot of runs in a row. Try again in a few minutes.", 429);
  recent.push(now); hits.set(ip, recent);
}

const isId = (s) => /^[0-9a-f-]{36}$/i.test(String(s || ""));

function send(res, fn) {
  res.setHeader("Cache-Control", "no-store");
  return fn().then((b) => res.status(200).json(b)).catch((e) => res.status(e.code || 500).json({ error: e.message }));
}

module.exports = { keyOf, runway, decide, ensureRouter, clean, imagePrompt, variantOf, SCENES, videoPrompt, limit, isId, send };
