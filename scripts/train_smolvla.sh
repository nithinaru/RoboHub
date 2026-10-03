#!/bin/zsh
# Fine-tune SmolVLA (lerobot/smolvla_base) on RoboHub's LeRobot dataset with the stock lerobot-train, recorded
# verbatim for the film by scripts/termrec.py (data/term/smolvla-$RUN.jsonl). Lowest priority: the research render runs.
#   observation.images.front (our sim camera) -> observation.images.camera1 (smolvla_base's first camera slot)
set -e
cd ${0:A:h}/..
export PATH="$PWD/.venv/bin:$PATH" HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1
STEPS=${STEPS:-650}
RUN=${RUN:-robohub}   # 650 steps -> data/smolvla/robohub (39 min); STEPS=1000 RUN=robohub-1h (about an hour)
exec .venv/bin/python scripts/termrec.py data/term/smolvla-$RUN.jsonl -- nice -n 19 taskpolicy -b lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/rohub_put_the_red_block_in_the_bowl \
  --dataset.root=data/web-runs/put-the-red-block-in-the-bowl/lerobot \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --batch_size=8 --steps=$STEPS --log_freq=25 \
  --policy.device=mps --policy.push_to_hub=false --wandb.enable=false \
  --output_dir=data/smolvla/$RUN --job_name=smolvla_$RUN
