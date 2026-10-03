// POST {prompt, budget, variant} with X-Gemini-Key -> Gemini draws the first frame and returns it inline.
const { keyOf, still, decide, clean, imagePrompt, variantOf, limit, send } = require("../_lib");

module.exports = (req, res) => send(res, async () => {
  if (req.method !== "POST") throw Object.assign(new Error("POST only"), { code: 405 });
  const key = keyOf(req);
  const body = req.body || {};
  const task = clean(body.prompt);
  const variant = variantOf(body.variant);
  limit(req);
  const frame = await still(key, imagePrompt(task, variant));
  return { task, ...decide(task, body.budget), variant, image_b64: frame.b64, mime: frame.mime, image_model: frame.model };
});
