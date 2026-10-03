// GET ?uri=<Gemini file URI> with X-Gemini-Key -> the mp4 bytes. The browser cannot send the key on a <video> tag.
const { keyOf } = require("../_lib");

module.exports = async (req, res) => {
  try {
    const key = keyOf(req);
    const uri = String((req.query || {}).uri || "");
    if (!uri.startsWith("https://generativelanguage.googleapis.com/")) {
      res.status(400).json({ error: "bad video uri" });
      return;
    }
    const r = await fetch(uri, { headers: { "x-goog-api-key": key }, redirect: "follow" });
    if (!r.ok) {
      res.status(502).json({ error: `Gemini video download failed (${r.status})` });
      return;
    }
    const buf = Buffer.from(await r.arrayBuffer());
    res.setHeader("Content-Type", r.headers.get("content-type") || "video/mp4");
    res.setHeader("Cache-Control", "no-store");
    res.status(200).send(buf);
  } catch (e) {
    res.status(e.code || 500).json({ error: e.message });
  }
};
