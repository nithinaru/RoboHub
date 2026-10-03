#!/bin/zsh
# Build pybullet 3.2.7 for macOS arm64 / Python 3.12 (no wheel on PyPI; the sdist fails on the macOS 15 SDK because
# examples/ThirdPartyLibs/zlib/zutil.h defines fdopen(fd, mode) as NULL whenever TARGET_OS_MAC is set, which
# collides with the real fdopen in <stdio.h>). Removes that one define and builds data/wheels/pybullet-*.whl.
set -e
cd ${0:A:h}/..
W=$(mktemp -d)
curl -sL -o $W/pybullet.tar.gz "$(curl -s https://pypi.org/pypi/pybullet/3.2.7/json | python3 -c 'import json,sys;print([u["url"] for u in json.load(sys.stdin)["urls"] if u["packagetype"]=="sdist"][0])')"
tar xzf $W/pybullet.tar.gz -C $W
python3 - $W/pybullet-3.2.7/examples/ThirdPartyLibs/zlib/zutil.h <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "#ifndef fdopen\n#define fdopen(fd, mode) NULL /* No fdopen() */\n#endif\n"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, "/* fdopen exists on macOS (patched) */\n"))
PY
mkdir -p data/wheels
nice -n 19 uv build --wheel --python .venv/bin/python -o data/wheels $W/pybullet-3.2.7
