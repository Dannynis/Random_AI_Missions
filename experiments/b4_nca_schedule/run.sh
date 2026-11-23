#!/bin/bash
# B4: persistent NCA re-seed schedules, 6 arms x 10 seeds (4 runs share one GPU).
# Resumable: each (arm, seed) writes its checkpoint and result JSON to $RUN_DIR/ckpt.
# usage: run.sh [tiny]
set -euo pipefail
cd "$(dirname "$0")"
export LAB_DATA=${LAB_DATA:-$HF_HOME/lab_data}
[ "${1:-}" = "tiny" ] && export NCA_TINY=1
N=${NCA_WORKERS:-$(nproc)}
pids=()
for w in $(seq 0 $((N - 1))); do python b4_bench.py worker $w $N & pids+=($!); done
trap 'kill ${pids[*]} 2>/dev/null' USR1 TERM
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
python b4_bench.py collect
exit $rc
