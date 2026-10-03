#!/bin/zsh
# Replay every dataset episode in MuJoCo and in PyBullet (recorded verbatim to data/term/crosscheck.jsonl), then
# film one episode in both engines from the same viewpoint.
set -e
cd ${0:A:h}/..
export PYTHONPATH=src
.venv/bin/python scripts/termrec.py data/term/crosscheck.jsonl -- nice -n 19 taskpolicy -b .venv/bin/python -m rohub.crosscheck data/web-runs/put-the-red-block-in-the-bowl
nice -n 19 taskpolicy -b .venv/bin/python -m rohub.crosscheck data/web-runs/put-the-red-block-in-the-bowl --film ${FILM_EP:-0}
