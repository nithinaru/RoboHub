// GET ?id=<Veo operation name> with X-Gemini-Key -> {status, progress, output, failure}
const { keyOf, google, operationStatus, isOp, send } = require("../_lib");

module.exports = (req, res) => send(res, async () => {
  const key = keyOf(req);
  const id = (req.query || {}).id;
  if (!isOp(id)) throw Object.assign(new Error("bad task id"), { code: 400 });
  const j = await google(key, id);
  return operationStatus(j);
});
