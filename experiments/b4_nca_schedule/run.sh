#!/bin/bash
# B4: persistent NCA re-seed schedules, 6 arms x 5 seeds (CPU workers; tiny model).
# Resumable: each (arm, seed) writes its checkpoint and result JSON to $RUN_DIR/ckpt.
# usage: run.sh [tiny]
set -euo pipefail
cd "$(dirname "$0")"
export LAB_DATA=${LAB_DATA:-$HF_HOME/lab_data}
[ "${1:-}" = "tiny" ] && export NCA_TINY=1
export NCA_WORKERS=${NCA_WORKERS:-$(nproc)}
python b4_bench.py all
