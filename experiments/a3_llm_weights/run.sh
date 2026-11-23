#!/bin/bash
# A3: LLM-generated network weights with a real open LLM (LoRA). Resumable: finished arms are skipped and an
# unfinished training arm resumes from $RUN_DIR/ckpt. Pass --tiny for a quick validation run.
# A3_PYTHON may point at an interpreter that also has `tabpfn` installed; defaults to `python`.
set -euo pipefail
cd "$(dirname "$0")"
export USE_TF=0 TRANSFORMERS_NO_TF=1 TOKENIZERS_PARALLELISM=false
# one BLAS thread per process: the task-sampler workers otherwise oversubscribe the allocated cores
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${A3_PYTHON:-python}" -u a3/main.py "$@"
