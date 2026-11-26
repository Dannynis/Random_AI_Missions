#!/bin/bash
# B1: layer-local training vs backprop with a depth sweep on CIFAR-10/100.
# Three rules (bp, local_ce, ff) x depths x seeds; several cells share one GPU.
# Resumable: each (rule, dataset, depth, seed) cell checkpoints to $RUN_DIR/ckpt
# at epoch level and writes its metrics JSON there when done; summary -> out/.
# usage: run.sh [tiny]
set -euo pipefail
cd "$(dirname "$0")"
export LAB_DATA=${LAB_DATA:-$HF_HOME/lab_data}
[ "${1:-}" = "tiny" ] && export B1_TINY=1
N=${B1_WORKERS:-3}
pids=()
for w in $(seq 0 $((N - 1))); do python b1_depth.py worker "$w" "$N" & pids+=($!); done
trap 'kill -USR1 ${pids[*]} 2>/dev/null' USR1
trap 'kill ${pids[*]} 2>/dev/null' TERM
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
python b1_depth.py collect
exit $rc
