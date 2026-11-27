#!/bin/bash
# B8: exemplar selection for a nearest-neighbour memory classifier on frozen DINOv2 features,
# class-incremental CIFAR-100 (10 stages) and CIFAR-10 (5 stages), 10 seeds.
# Features come from the A2 extractor's shared cache ($LAB_DATA/features; extracted if missing).
# Resumable: one JSON per (dataset, feature, seed) in $RUN_DIR/ckpt; summary -> $RUN_DIR/out.
# usage: run.sh [tiny]
set -euo pipefail
cd "$(dirname "$0")"
export LAB_DATA=${LAB_DATA:-$HF_HOME/lab_data}
[ "${1:-}" = "tiny" ] && export B8_TINY=1
FEATS=${B8_FEATS:-"dinov2-small dinov2-base dinov2-large"}
for m in $FEATS; do for d in cifar100 cifar10; do python ../a2_al_policy/extract.py $m $d; done; done
for m in $FEATS; do for d in cifar100 cifar10; do python b8_bench.py worker $d $m; done; done
python b8_bench.py collect
