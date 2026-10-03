"""VLA tile films for the site: the fine-tuned SmolVLA running each accepted clip's scenario in MuJoCo.
See src/rohub/vla_tiles.py for the two modes (rollout, --from-eval).

    PYTHONPATH=src python scripts/vla_tiles.py data/web-runs/put-the-red-block-in-the-bowl --ckpt <ckpt> --device cuda
    PYTHONPATH=src python scripts/vla_tiles.py data/web-runs/put-the-red-block-in-the-bowl --from-eval eval.json eval_traj.npz
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rohub.vla_tiles import main  # noqa: E402

if __name__ == "__main__":
    main()
