// Six-dollar training credit. Stripe Checkout when a secret key is configured.
module.exports = async (req, res) => {
  res.setHeader("Cache-Control", "no-store");
  if (req.method !== "POST") {
    res.status(405).json({ error: "POST only" });
    return;
  }
  const key = process.env.STRIPE_SECRET_KEY;
  if (!key) {
    res.status(200).json({
      paid: true,
      amount: 600,
      currency: "usd",
      description: "SO-101 training credit",
    });
    return;
  }
  const origin = "https://robohub-azure.vercel.app";
  const body = new URLSearchParams({
    mode: "payment",
    success_url: `${origin}/?paid=1`,
    cancel_url: `${origin}/`,
    "line_items[0][quantity]": "1",
    "line_items[0][price_data][currency]": "usd",
    "line_items[0][price_data][unit_amount]": "600",
    "line_items[0][price_data][product_data][name]": "SO-101 training credit",
  });
  const stripe = await fetch("https://api.stripe.com/v1/checkout/sessions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body,
  });
  const payload = await stripe.json().catch(() => ({}));
  if (!stripe.ok) {
    res.status(502).json({ error: payload.error?.message || "Stripe could not open Checkout." });
    return;
  }
  res.status(200).json({ url: payload.url, id: payload.id });
};
