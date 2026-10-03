// GET ?uri=<Gemini file URI> with X-Gemini-Key -> the mp4 bytes. The browser cannot send the key on a <video> tag.
const { keyOf, allowedMediaUri } = require("../_lib");

function allowedHop(uri) {
  let url;
  try { url = new URL(uri); } catch { return false; }
  const host = url.hostname;
  const google = host === "generativelanguage.googleapis.com"
    || host === "storage.googleapis.com"
    || host.endsWith(".googleusercontent.com");
  return url.protocol === "https:" && !url.username && !url.password && google && !url.pathname.includes("..");
}

module.exports = async (req, res) => {
  try {
    const key = keyOf(req);
    const uri = String((req.query || {}).uri || "");
    if (!allowedMediaUri(uri)) {
      res.status(400).json({ error: "bad video uri" });
      return;
    }
    let r = await fetch(uri, { headers: { "x-goog-api-key": key }, redirect: "manual" });
    if (r.status >= 300 && r.status < 400) {
      const next = new URL(r.headers.get("location") || "", uri).href;
      if (!allowedHop(next)) {
        res.status(502).json({ error: "Gemini video redirect left Google" });
        return;
      }
      r = await fetch(next, { redirect: "manual" });
    }
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
