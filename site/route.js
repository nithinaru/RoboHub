// Budget routing: every run goes through a quality-optimized Runway Model Router with a price ceiling.
// You pick what a run may cost ($2 to $10 for its 5 demonstration clips); the router then picks the best model
// that fits under (budget / 5 clips) credits a clip. 1 credit = $0.01.
//   Route.decide(text, {budget}) -> {router: "robohub-q<ceiling>", budget, ceiling, reasons: [...]}
(() => {
  const CLIPS = 5;
  const BUDGETS = [2, 4, 6, 8, 10];
  const ceiling = (usd) => Math.round((usd * 100) / CLIPS);
  let budget = 2;

  function decide(_text, opts = {}) {
    const usd = BUDGETS.includes(opts.budget) ? opts.budget : budget;
    const c = ceiling(usd);
    return { router: `robohub-q${c}`, budget: usd, ceiling: c, reasons: [`best quality up to ${c} credits a clip`] };
  }

  const NAME = { "demo-cheap": "$2", "demo-fast": "Fast", "demo-best": "$10" };
  for (const b of BUDGETS) NAME[`robohub-q${ceiling(b)}`] = `$${b}`;
  window.Route = {
    decide, NAME, BUDGETS, CLIPS, ceiling,
    get budget() { return budget; },
    set budget(b) { if (BUDGETS.includes(b)) budget = b; },
  };
})();
