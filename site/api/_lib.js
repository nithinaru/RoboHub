// Gemini lead: a still from Gemini, then Veo image-to-video. The visitor's key is forwarded to Google and never stored.
const BASE = "https://generativelanguage.googleapis.com/v1beta";
const IMAGE_MODEL = "gemini-3.1-flash-image-preview";
const VEO_FAST = "veo-3.1-fast-generate-preview";
const VEO = "veo-3.1-generate-preview";
const PER_IP = 12;
const hits = new Map();

const fail = (msg, code) => Object.assign(new Error(msg), { code });

function keyOf(req) {
  let raw = req.headers["x-gemini-key"] || "";
  if (Array.isArray(raw)) raw = raw[0] || "";
  const k = String(raw).trim();
  if (!k) throw fail("Add your Gemini API key to generate new prompts.", 401);
  if (!/^(?:AIza|AQ\.)[A-Za-z0-9_-]{20,}$/.test(k)) throw fail("That doesn't look like a Gemini API key (it starts with AIza or AQ.).", 401);
  return k;
}

function veoModel(budget) {
  const n = Number(budget);
  if (!Number.isFinite(n)) return VEO_FAST;
  return n >= 6 ? VEO : VEO_FAST;
}

function nearestBudget(budget) {
  const n = Number(budget);
  if (!Number.isFinite(n)) return 2;
  return BUDGETS.reduce((best, b) => (Math.abs(b - n) < Math.abs(best - n) ? b : best), 2);
}

async function google(key, path, { method = "GET", body } = {}) {
  const r = await fetch(path.startsWith("http") ? path : `${BASE}/${path.replace(/^\//, "")}`, {
    method,
    headers: { "x-goog-api-key": key, "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const j = await r.json().catch(() => ({}));
  if (r.status === 401 || r.status === 403) throw fail("Gemini rejected this API key.", 401);
  if (!r.ok) {
    const code = r.status === 400 || r.status === 429 ? r.status : 502;
    throw fail(j.error?.message || `Gemini HTTP ${r.status}`, code);
  }
  return j;
}

async function still(key, prompt) {
  const j = await google(key, `models/${IMAGE_MODEL}:generateContent`, {
    method: "POST",
    body: {
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { responseModalities: ["IMAGE"] },
    },
  });
  const parts = j.candidates?.[0]?.content?.parts || [];
  const inline = parts.map((p) => p.inlineData || p.inline_data).find(Boolean);
  if (!inline?.data) throw fail("Gemini returned no first frame.", 502);
  return { b64: inline.data, mime: inline.mimeType || inline.mime_type || "image/png", model: IMAGE_MODEL };
}

async function startVideo(key, { prompt, b64, mime, budget }) {
  const model = veoModel(budget);
  const j = await google(key, `models/${model}:predictLongRunning`, {
    method: "POST",
    body: {
      instances: [{
        prompt,
        image: { inlineData: { mimeType: mime || "image/png", data: b64 } },
      }],
      parameters: { aspectRatio: "16:9", durationSeconds: 8 },
    },
  });
  if (!isOp(j.name)) throw fail("Veo did not start a video job.", 502);
  return { name: j.name, model };
}

function videoUri(j) {
  const box = j.response?.generateVideoResponse || {};
  const samples = box.generatedSamples || box.videos || [];
  return samples[0]?.video?.uri || samples[0]?.uri || null;
}

function operationStatus(j) {
  if (!j || typeof j !== "object") return { status: "FAILED", failure: "Empty Veo status.", output: null, progress: null };
  if (j.error) return { status: "FAILED", failure: j.error.message || "Veo failed", output: null, progress: null };
  if (!j.done) return { status: "RUNNING", progress: 0.5, output: null, failure: null };
  const uri = videoUri(j);
  if (!allowedMediaUri(uri)) {
    const why = j.response?.generateVideoResponse?.raiMediaFilteredReasons?.[0] || "Veo finished without a video.";
    return { status: "FAILED", failure: why, output: null, progress: 1 };
  }
  return { status: "SUCCEEDED", progress: 1, output: [uri], failure: null };
}

function allowedMediaUri(uri) {
  let url;
  try { url = new URL(String(uri || "")); } catch { return false; }
  return url.protocol === "https:"
    && url.hostname === "generativelanguage.googleapis.com"
    && !url.username
    && !url.password
    && !url.pathname.includes("..");
}

const BUDGETS = [2, 4, 6, 8, 10];
function decide(_task, budget) {
  const usd = nearestBudget(budget);
  return { budget: usd, model: veoModel(usd), reasons: [`Gemini ${veoModel(usd)}`] };
}

function clean(prompt) {
  const p = String(prompt ?? "").replace(/\s+/g, " ").trim().replace(/[.!?,;:]+$/g, "");
  if (p.length < 6 || p.length > 140 || !/^[a-zA-Z0-9 ,'-]+$/.test(p)) {
    throw fail("Use a short plain sentence (6 to 140 letters), like \"put the green cup on the plate\".", 400);
  }
  return p;
}

const SCENES = [
  { table: "a plain light oak wooden tabletop", light: "Soft natural daylight from a window on the left", angle: "three-quarter view from slightly above" },
  { table: "a matte white laminate desk", light: "Warm late-afternoon sunlight with long soft shadows", angle: "front view from slightly to the left, about 35 degrees down" },
  { table: "a dark walnut table", light: "Bright even studio light", angle: "low front view from table height, about 25 degrees down" },
];
function variantOf(v) {
  if (v == null || v === "") return 0;
  if (typeof v !== "number" && typeof v !== "string") throw fail(`variant must be 0 to ${SCENES.length - 1}`, 400);
  if (typeof v === "string" && !/^\d+$/.test(v)) throw fail(`variant must be 0 to ${SCENES.length - 1}`, 400);
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

const isOp = (s) => {
  const text = String(s || "");
  return !text.includes("..") && /^models\/[A-Za-z0-9._-]+\/operations\/[A-Za-z0-9_-]+$/.test(text);
};

function send(res, fn) {
  res.setHeader("Cache-Control", "no-store");
  return fn().then((b) => res.status(200).json(b)).catch((e) => res.status(e.code || 500).json({ error: e.message }));
}

module.exports = {
  keyOf, google, still, startVideo, operationStatus, veoModel, nearestBudget, decide, clean,
  imagePrompt, variantOf, videoPrompt, limit, isOp, allowedMediaUri, send, IMAGE_MODEL,
};
