#!/bin/zsh
# Evaluate the fine-tuned SmolVLA in the MuJoCo harness on the MLP's 50 unseen cube positions (recorded verbatim to
# data/term/smolvla-$RUN-eval.jsonl), then film the first successful evaluated episode at 1920x1080 (asserted identical).
set -e
cd ${0:A:h}/..
export HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src
RUN=${RUN:-robohub}
CK=data/smolvla/$RUN/checkpoints/last/pretrained_model
.venv/bin/python scripts/termrec.py data/term/smolvla-$RUN-eval.jsonl -- nice -n 19 taskpolicy -b .venv/bin/python -m rohub.vla $CK data/web-runs/put-the-red-block-in-the-bowl
nice -n 19 taskpolicy -b .venv/bin/python -m rohub.vla $CK data/web-runs/put-the-red-block-in-the-bowl --hero
