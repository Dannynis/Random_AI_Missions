#!/bin/bash
# A1 entry point. usage: run.sh [tiny|full|eval] ; env SEEDS (default "0 1 2 3 4"), STEPS (default 40000).
# Resumable: training checkpoints in $RUN_DIR/ckpt, per-seed evaluation results in $RUN_DIR/out are skipped when complete.
set -euo pipefail
cd "$(dirname "$0")"
MODE=${1:-full}
python prep_data.py
if [ "$MODE" = tiny ]; then
  python run_a1.py train --seed 0 --steps 300 --batch 8 --ckpt_every 100
  python run_a1.py eval --seed 0 --tiny --conditions clean ota mp3_64k --pesq_targets 4.0
  python run_a1.py summary
  exit 0
fi
if [ "$MODE" = speed ]; then
  python run_a1.py train --seed 99 --steps 400 --batch 32 --ckpt_every 200
  exit 0
fi
SEEDS=${SEEDS:-"0 1 2 3 4"}; STEPS=${STEPS:-40000}
if [ "$MODE" = stress ]; then             # evaluation only, unseen channels; models from A1_CKPT
  for s in $SEEDS; do
    python run_a1.py eval --seed "$s" --tag _stress --conditions ota_noisy speed_1pct phone opus_12k aac_32k
  done
  python run_a1.py summary
  exit 0
fi
if [ "$MODE" = ablate_channel ]; then      # ours only, trained on the original simple channel
  for s in $SEEDS; do
    python run_a1.py train --seed "$s" --steps "$STEPS" --train_channel legacy
    python run_a1.py eval --seed "$s" --methods ours
    python run_a1.py summary
  done
  exit 0
fi
for s in $SEEDS; do
  python run_a1.py train --seed "$s" --steps "$STEPS"
  python run_a1.py eval --seed "$s"
  python run_a1.py summary
done
