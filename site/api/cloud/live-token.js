// Short-lived token for the browser Live socket. The project key stays on the server.
const { send } = require("../_lib");

module.exports = (req, res) => send(res, async () => {
  const key = process.env.GEMINI_API_KEY;
  if (!key) throw Object.assign(new Error("Gemini is not configured on the server."), { code: 503 });
  const r = await fetch("https://generativelanguage.googleapis.com/v1beta/auth_tokens", {
    method: "POST",
    headers: { "x-goog-api-key": key, "Content-Type": "application/json" },
    body: "{}",
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok || !j.name) {
    throw Object.assign(new Error(j.error?.message || "Could not start a voice session."), { code: 502 });
  }
  return { token: j.name };
});
