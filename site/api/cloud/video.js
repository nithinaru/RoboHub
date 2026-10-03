// POST {prompt, image_task} with X-Runway-Key -> the visitor's Model Router picks the video model and animates the frame.
const { keyOf, runway, decide, ensureRouter, clean, videoPrompt, isId, limit, send } = require("../_lib");

module.exports = (req, res) => send(res, async () => {
  if (req.method !== "POST") throw Object.assign(new Error("POST only"), { code: 405 });
  const key = keyOf(req);
  const { prompt, image_task, budget } = req.body || {};
  const task = clean(prompt);
  if (!isId(image_task)) throw Object.assign(new Error("bad image task"), { code: 400 });
  limit(req);
  const img = await runway(key, "GET", `/tasks/${image_task}`);
  if (img.status !== "SUCCEEDED" || !img.output || !img.output[0]) throw Object.assign(new Error("first frame is not ready"), { code: 409 });
  const { router, ceiling } = decide(task, budget);
  const configId = await ensureRouter(key, router, ceiling);
  const t = await runway(key, "POST", "/generate/video", {
    configId,
    input: { promptText: videoPrompt(task), aspectRatio: "16:9", duration: 5, referenceImages: [{ uri: img.output[0], role: "first" }] },
  });
  const r = t.routing || {};
  return { video_task: t.id, router, configId, model: r.model || null, credits: (r.estimatedCost || {}).credits ?? null };
});
