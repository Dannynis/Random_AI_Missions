#!/bin/bash
# A2: cold-start active learning on frozen DINOv2 features (CIFAR-10 / CIFAR-100).
# usage: run.sh tiny | baselines [feat] | policy [feat] | policy2 [feat]
#   features are extracted first if missing (shared cache in $LAB_DATA/features, atomic writes);
#   per-(task, method, seed) results and the policy checkpoint live in $RUN_DIR/ckpt (resumable).
set -euo pipefail
cd "$(dirname "$0")"
export LAB_DATA=${LAB_DATA:-$HF_HOME/lab_data}
MODE=${1:-baselines}; FEATM=${2:-dinov2-base}
if [ "$MODE" = tiny ]; then
  export AL_TINY=1
  for d in cifar10 cifar100; do python extract.py dinov2-small $d --tiny; done
  python al_bench.py baselines dinov2-small
  python al_bench.py policy dinov2-small
  python al_bench.py policy2 dinov2-small
  exit 0
fi
for m in "$FEATM" dinov2-small; do for d in cifar10 cifar100; do python extract.py $m $d; done; done
python al_bench.py "$MODE" "$FEATM"
