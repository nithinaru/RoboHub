#!/bin/zsh
# BEFORE number, "out of the box" variant: lerobot/smolvla_base weights AND its own processors (the SO-100
# pretraining mean/std for state and action), our camera renamed to camera1. Same harness, seeds, success rule.
set -e
cd ${0:A:h}/..
export HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src
CK=data/smolvla/robohub-1h/checkpoints/last/pretrained_model
BASE=$(ls -d ~/.cache/huggingface/hub/models--lerobot--smolvla_base/snapshots/*/ | head -1)
OUT=data/smolvla/base-ootb
exec .venv/bin/python scripts/termrec.py data/term/smolvla-base-ootb-eval.jsonl -- nice -n 19 taskpolicy -b \
  .venv/bin/python -m rohub.vla $CK data/web-runs/put-the-red-block-in-the-bowl \
  --weights lerobot/smolvla_base --proc $BASE --out $OUT/eval.json
