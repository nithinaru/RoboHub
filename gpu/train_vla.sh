#!/bin/bash
# usage: train_vla.sh <lerobot_dataset_root> <steps> [extra rohub_trainer.py args]
# Trains SmolVLA on a rented RunPod GPU and streams [gpu]/[train]/[eval]/[cost] lines.
exec /usr/bin/env python3 "$(dirname "$0")/rohub_trainer.py" "$@"
