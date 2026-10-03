// Tells the page that the server holds a Gemini key. Never returns the key.
module.exports = (req, res) => {
  res.setHeader("Cache-Control", "no-store");
  res.status(200).json({ gemini: Boolean(process.env.GEMINI_API_KEY) });
};
