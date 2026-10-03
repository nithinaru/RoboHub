#!/bin/zsh
# Stops the P4 training PID given (started by this session's probe chain) if free disk drops below 8 GiB
# (Jarvis, 2026-09-26: the research render auto-pauses at 5). Logs the step reached. Kills only that PID.
cd ${0:A:h}/..
T=$1 LOG=${2:-data/probes/train-aug-3000.log}
while kill -0 $T 2>/dev/null; do
  f=$(df -g / | awk 'NR==2{print $4}')
  if (( f < 8 )); then
    step=$(tail -c 3000 $LOG | tr '\r' '\n' | grep -o 'step:[0-9K]*' | tail -1)
    kill -TERM $T
    print "[guard $(date +%H:%M:%S)] free disk ${f} GiB < 8: stopped training pid $T at $step" | tee -a data/probes/chain.log
    exit 0
  fi
  sleep 180
done
print "[guard $(date +%H:%M:%S)] training pid $T ended on its own"
