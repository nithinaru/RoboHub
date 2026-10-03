// POST {prompt, image_b64, mime, budget} with X-Gemini-Key -> Veo animates the still. Returns the operation name.
const { keyOf, startVideo, clean, videoPrompt, limit, send } = require("../_lib");

module.exports = (req, res) => send(res, async () => {
  if (req.method !== "POST") throw Object.assign(new Error("POST only"), { code: 405 });
  const key = keyOf(req);
  const { prompt, image_b64, mime, budget } = req.body || {};
  const task = clean(prompt);
  if (!image_b64 || String(image_b64).length < 100) throw Object.assign(new Error("first frame is missing"), { code: 400 });
  limit(req);
  const job = await startVideo(key, { prompt: videoPrompt(task), b64: image_b64, mime, budget });
  return { video_task: job.name, model: job.model };
});
