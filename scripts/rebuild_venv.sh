#!/bin/zsh
# Rebuild .venv from uv.lock (it was removed by a disk cleanup on 2026-09-27 01:39). A large download is a heavy step:
# take ~/helloworld/.heavy-lock by the CLAUDE.md protocol, nice 19 + taskpolicy -b, release, then restart the probe chain.
cd ${0:A:h}/..
LOCK=$HOME/helloworld/.heavy-lock
log() { print -r -- "[venv $(date +%H:%M:%S)] $*" }
release() { [[ -f $LOCK/owner && "$(cat $LOCK/owner)" == "$$" ]] && rm -r $LOCK; true }
trap release EXIT INT TERM
while true; do
  f=$(df -g / | awk 'NR==2{print $4}'); s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); printf "%d", $6/1024}')
  if (( f < 15 || s > 10 )); then log "waiting: $f GiB, swap $s GB"; sleep 300; continue; fi
  if mkdir $LOCK 2>/dev/null; then print $$ > $LOCK/owner; break; fi
  o=$(cat $LOCK/owner 2>/dev/null)
  if [[ -n $o ]] && ! kill -0 $o 2>/dev/null; then log "stale lock ($o), taking it"; rm -r $LOCK; continue; fi
  sleep 60
done
log "uv sync (frozen, from uv.lock)"
nice -n 19 taskpolicy -b uv sync --frozen && log "done: $(du -sh .venv | cut -f1), $(df -g / | awk 'NR==2{print $4}') GiB free" || { log "uv sync FAILED"; exit 1; }
.venv/bin/python -c "import lerobot, torch, mujoco; print('imports ok', lerobot.__version__, torch.__version__)" 2>&1 | grep -v objc | tail -1
release
nohup scripts/probe_chain.sh >> data/probes/chain.log 2>&1 &
print $! > data/probes/chain.pid; log "chain restarted pid $(cat data/probes/chain.pid)"
