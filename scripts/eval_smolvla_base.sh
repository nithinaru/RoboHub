#!/bin/zsh
# BEFORE number for v8: lerobot/smolvla_base with NO fine-tuning, in the same MuJoCo harness, on the same 50 unseen
# cube positions and success rule as the fine-tuned eval (scripts/eval_smolvla.sh). The processors (rename map
# front -> camera1, our dataset's mean/std) come from the fine-tuned checkpoint, i.e. exactly what lerobot-train
# builds at step 0; the only difference from the AFTER model is the 1000 gradient steps.
set -e
cd ${0:A:h}/..
export HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src
CK=data/smolvla/robohub-1h/checkpoints/last/pretrained_model
OUT=data/smolvla/base-zeroshot
exec .venv/bin/python scripts/termrec.py data/term/smolvla-base-zeroshot-eval.jsonl -- nice -n 19 taskpolicy -b \
  .venv/bin/python -m rohub.vla $CK data/web-runs/put-the-red-block-in-the-bowl \
  --weights lerobot/smolvla_base --out $OUT/eval.json
