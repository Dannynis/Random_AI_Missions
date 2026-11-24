#!/bin/bash
# B7 step 1: baseline active-learning rules on hard pools (long-tail + label noise),
# reusing the A2 library and its cached DINOv2 features.
# usage: run.sh tiny | baselines [feat]
#   features are extracted via the A2 extractor into the shared $LAB_DATA/features cache (reused if present);
#   per-(task, method, seed) results live in $RUN_DIR/ckpt (resumable); summary in $RUN_DIR/out.
set -euo pipefail
cd "$(dirname "$0")"
export LAB_DATA=${LAB_DATA:-$HF_HOME/lab_data}
export PYTHONPATH="$(cd ../a2_al_policy && pwd):${PYTHONPATH:-}"
MODE=${1:-baselines}; FEATM=${2:-dinov2-base}
A2=../a2_al_policy
if [ "$MODE" = tiny ]; then
  export AL_TINY=1
  for d in cifar10 cifar100; do (cd "$A2" && python extract.py dinov2-small "$d" --tiny); done
  python b7_bench.py baselines dinov2-small
  python b7_bench.py collect
  exit 0
fi
for d in cifar10 cifar100; do (cd "$A2" && python extract.py "$FEATM" "$d"); done
python b7_bench.py "$MODE" "$FEATM"
python b7_bench.py collect
