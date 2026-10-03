#!/bin/zsh
# The small MLP baseline on the site's dataset, in a terminal (recorded verbatim to data/term/mlp.jsonl).
set -e
cd ${0:A:h}/..
exec bin/robohub train put-the-red-block-in-the-bowl
