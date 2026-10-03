// POST {prompt, budget, variant} with X-Runway-Key -> route the sentence, then Runway draws the first frame (gen4_image).
//   variant 0..2 picks one of 3 scene restyles (table, light, angle) so a run's 3 clips differ.
const { keyOf, runway, decide, clean, imagePrompt, variantOf, limit, send } = require("../_lib");

module.exports = (req, res) => send(res, async () => {
  if (req.method !== "POST") throw Object.assign(new Error("POST only"), { code: 405 });
  const key = keyOf(req);
  const body = req.body || {};
  const task = clean(body.prompt);
  const variant = variantOf(body.variant);
  limit(req);
  const t = await runway(key, "POST", "/text_to_image", { model: "gen4_image", promptText: imagePrompt(task, variant), ratio: "1280:720" });
  return { task, ...decide(task, body.budget), variant, image_task: t.id };
});
