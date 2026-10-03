"""lerobot-train, unchanged, except the checkpoint keeps only pretrained_model/ (no optimizer/scheduler state).
Probe runs are never resumed, and the Mac is ~2 GB above its 20 GB free-disk floor: this saves ~400 MB per run."""

import lerobot.scripts.lerobot_train as T

_orig = T.save_checkpoint
T.save_checkpoint = lambda **kw: _orig(**{**kw, "optimizer": None, "scheduler": None})

if __name__ == "__main__":
    T.main()
