"""Pre-render the hosted site's preview films (deploy-v2/previews/*.mp4 + manifest.json): one scripted MuJoCo
preview per primitive family, picked on Vercel by the keyword rules in site.js (pvKind). Run from the repo root:
  nice -n 19 taskpolicy -b .venv/bin/python scripts/preview_set.py
Takes ~/helloworld/.heavy-lock for the batch; each film renders in its own process."""
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rohub.vla_tiles import heavy_lock
from rohub import preview as P
SET = [
    ("pick-bowl", "put the red block in the bowl"),
    ("pick-plate", "put the red block on the plate"),
    ("pick-cup", "put the ball in the cup"),
    ("pick-box", "put the blue block in the box"),
    ("pick-pad", "place the red block on the blue square"),
    ("push-pad", "push the green cube onto the blue square"),
    ("stack", "stack the yellow block on the red block"),
    ("tower", "build a tower of three blocks"),
    ("lift", "pick up the red block"),
    ("hinge-open", "open the book"),
    ("hinge-close", "close the book"),
]
out = Path("deploy-v2/previews"); man = []
with heavy_lock(str(Path.home() / "helloworld/.heavy-lock")):
    for name, s in SET:
        plan = P.parse(s)
        import subprocess
        pf = out / f"{name}.plan.json"; pf.write_text(json.dumps(plan))
        r = subprocess.run([sys.executable, "-m", "rohub.preview", "--plan", str(pf), "--out", str(out / f"{name}.mp4")],
                           capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"}, check=True)
        st = json.loads(r.stdout.strip().splitlines()[-1]); pf.unlink(); (out / f"{name}.json").unlink()
        man.append({"id": name, "file": f"previews/{name}.mp4", "family": plan["family"],
                    "target": next((o["kind"] for o in plan["objects"] if any(x.get("target") == o["name"] for x in plan["steps"])), None),
                    "movable": plan["objects"][0]["kind"] if plan["family"] not in ("hinge_open", "hinge_close") else None,
                    "summary": P.caption(plan), "example": s})
        print(name, st["render_s"], st["bytes"], st["max_ik_err"], flush=True)
(out / "manifest.json").write_text(json.dumps({"caption": P.CAPTION, "previews": man}, indent=1))
