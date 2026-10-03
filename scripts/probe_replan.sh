#!/bin/zsh
# Probe (v8 research, 0 credits, no training): the SAME 1000-step SmolVLA checkpoint, same 50 seeds and success rule,
# but replanning every N frames instead of running each 50-action chunk (1.67 s) open loop. N=${N:-10}.
set -e
cd ${0:A:h}/..
export HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src
N=${N:-10}
CK=data/smolvla/robohub-1h/checkpoints/last/pretrained_model
exec nice -n 19 taskpolicy -b .venv/bin/python -m rohub.vla $CK data/web-runs/put-the-red-block-in-the-bowl \
  --n-action-steps $N --out data/smolvla/probe-replan$N/eval.json
