#!/bin/zsh
# v8 research probes, 0 Runway credits. Every heavy step (training, eval) takes the shared ~/helloworld/.heavy-lock
# (CLAUDE.md, 2026-09-26: one heavy job at a time besides the research render; owner PID in .heavy-lock/owner;
# a dead owner's lock is stale and is taken over), runs at nice 19 + taskpolicy -b, and waits while swap > 10 GB or
# free disk < 15 GiB (evaluation-only steps, ~2 MB of output: 12 GiB). Steps already done are skipped, so the chain can be re-run after any stop.
# Order (2026-09-26, disk has room for ~1 more checkpoint): P2 eval (aug25, front, 1000 steps; trained 09-25)
# -> P4: aug25 front at 3000 steps (the steps lever, same data as P2) -> P3: aug25 front + wrist, 1000 steps.
set -e
cd ${0:A:h}/..
export PATH="$PWD/.venv/bin:$PATH" HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src
RUN=data/web-runs/put-the-red-block-in-the-bowl
AUG=data/probes/aug25
LOCK=$HOME/helloworld/.heavy-lock
LOW=(nice -n 19 taskpolicy -b)
log() { print -r -- "[chain $(date +%H:%M:%S)] $*" }
free_gib() { df -g / | awk 'NR==2{print $4}' }
swap_gb() { sysctl -n vm.swapusage | awk '{gsub("M","",$6); printf "%d", $6/1024}' }

release() { [[ -f $LOCK/owner && "$(cat $LOCK/owner)" == "$$" ]] && rm -r $LOCK; true }
trap release EXIT INT TERM

acquire() {  # wait for the machine and the lock; $1 = min free GiB (15 training, 12 eval-only per CLAUDE.md);
  # $2 = earliest HHMM of the no-start window (promised to the servo-metrology session 09-26: quiet 19:30-23:30 PT;
  # a ~2 h training must not start after 17:30)
  local need=${1:-15} quiet_from=${2:-1930}
  while true; do
    local f=$(free_gib) s=$(swap_gb) now=$(date +%H%M)
    if (( 10#$now >= 10#$quiet_from && 10#$now < 2330 )); then log "quiet window until 23:30"; sleep 300; continue; fi
    if (( f < need || s > 10 )); then log "waiting: ${f} GiB free, swap ${s} GB"; sleep 300; continue; fi
    if mkdir $LOCK 2>/dev/null; then print $$ > $LOCK/owner; return; fi
    local o=$(cat $LOCK/owner 2>/dev/null)
    if [[ -n $o ]] && ! kill -0 $o 2>/dev/null; then log "stale heavy-lock (pid $o dead), taking it"; rm -r $LOCK; continue; fi
    sleep 60
  done
}

train() {  # name steps rename_json
  local name=$1 steps=$2 ren=$3 out=data/probes/smolvla-$1
  [[ -f $out/checkpoints/last/pretrained_model/model.safetensors ]] && { log "$name already trained"; return; }
  acquire 15 1730; log "train $name ($steps steps)"
  $LOW .venv/bin/python scripts/lerobot_train_lean.py \
    --policy.path=lerobot/smolvla_base --dataset.repo_id=local/rohub_aug --dataset.root=$AUG \
    --dataset.video_backend=pyav --rename_map="$ren" \
    --batch_size=8 --steps=$steps --log_freq=25 --save_freq=$steps \
    --policy.device=mps --policy.push_to_hub=false --wandb.enable=false \
    --output_dir=$out --job_name=probe_$name > data/probes/train-$name.log 2>&1
  release; log "trained $name"
}
evaluate() {  # name cameras
  local ck=data/probes/smolvla-$1/checkpoints/last
  [[ -f $ck/eval.json ]] || {
    acquire 12; log "eval $1"
    $LOW .venv/bin/python -m rohub.vla $ck/pretrained_model $RUN --cameras $2 > data/probes/eval-$1.log 2>&1
    release
  }
  log "$1: $(python3 -c "import json;e=json.load(open('$ck/eval.json'));print(e['successes'],'/',e['seeds'],e['wilson95'])")"
}

[[ -f $AUG/augment.json ]] || { log "missing $AUG"; exit 2; }
F='{"observation.images.front": "observation.images.camera1"}'
FW='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
evaluate aug-front front
train aug-3000 3000 "$F"; evaluate aug-3000 front
hero() {  # 1920x1080 film of one evaluated P4 episode (re-drawn from its stored qpos), a few MB
  local ck=data/probes/smolvla-$1/checkpoints/last
  [[ -f $ck/hero.mp4 ]] && { log "$1 hero exists"; return; }
  acquire 15; log "hero $1"
  $LOW .venv/bin/python -m rohub.vla $ck/pretrained_model $RUN --hero > data/probes/hero-$1.log 2>&1
  release; log "hero $1: $(tail -1 data/probes/hero-$1.log)"
}
hero aug-3000
# P3 wrist dropped 09-26 (stalled under swap; lowest value). Eval-noise check of P4 instead: same checkpoint, same 50
# positions, flow-matching noise redrawn (torch seed + 1000). Eval-only, ~2 MB.
N=data/probes/smolvla-aug-3000/noise1000
if [[ ! -f $N/eval.json ]]; then
  acquire 12; log "eval aug-3000 noise+1000"
  $LOW .venv/bin/python -m rohub.vla data/probes/smolvla-aug-3000/checkpoints/last/pretrained_model $RUN \
    --noise-offset 1000 --out $N/eval.json > data/probes/eval-aug-3000-noise1000.log 2>&1
  release
fi
log "aug-3000 noise+1000: $(python3 -c "import json;e=json.load(open('$N/eval.json'));print(e['successes'],'/',e['seeds'],e['wilson95'])")"
log "all probes done"
