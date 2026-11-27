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
if [ "$MODE" = dear_tiny ]; then           # DeAR baseline smoke test: short training + tiny evaluation
  for s in ${SEEDS:-0}; do
    python dear.py train --seed "$s" --steps "${STEPS:-300}" --ckpt_every 100
    python run_a1.py eval --seed "$s" --methods dear --tag _dear --tiny --conditions clean ota --pesq_targets 4.0
  done
  exit 0
fi
if [ "$MODE" = dear_eval ]; then           # DeAR evaluation only, models from $A1_CKPT
  for s in ${SEEDS:-0 1 2}; do
    python run_a1.py eval --seed "$s" --methods dear --tag _dear
  done
  python run_a1.py summary
  exit 0
fi
if [ "$MODE" = dear_train ]; then         # DeAR baseline (dear.py): train the seeds side by side on the GPU, then the standard evaluation
  pids=""
  for s in ${SEEDS:-0 1 2}; do
    python dear.py train --seed "$s" --steps "${STEPS:-100000}" & pids="$pids $!"
  done
  for p in $pids; do wait "$p"; done
  for s in ${SEEDS:-0 1 2}; do
    python run_a1.py eval --seed "$s" --methods dear --tag _dear
  done
  python run_a1.py summary
  exit 0
fi
OPUS_CONDS="clean mp3_64k opus_32k reverb ota_legacy ota ota_mp3 ota_opus12"   # standard conditions + ota_opus12
if [ "$MODE" = opus_tiny ]; then           # codec-stage smoke test: same short training with and without the Opus stage
  root=$RUN_DIR
  for p in 0 ${OPUS_P:-0.5}; do          # one after the other, so s_per_step is a clean timing
    RUN_DIR=$root/out/p$p python run_a1.py train --seed 0 --steps "${TINY_STEPS:-4000}" --opus_p "$p" --ckpt_every 500
  done
  for p in 0 ${OPUS_P:-0.5}; do
    RUN_DIR=$root/out/p$p python run_a1.py eval --seed 0 --methods ours --no_native --pesq_targets 4.0 \
      --n_rec "${N_REC:-96}" --conditions clean ota ota_opus12
  done
  exit 0
fi
if [ "$MODE" = opus_full ]; then           # opus_tiny, a go / no-go check, then opus_train, in one job
  gate=$RUN_DIR/out/opus_gate.json
  if [ ! -f "$gate" ]; then
    bash run.sh opus_tiny
    python - "$RUN_DIR/out" "${OPUS_P:-0.5}" <<'PY'
import json, os, sys
d, p = sys.argv[1], sys.argv[2]
row = lambda q, c: next(r for r in json.load(open(os.path.join(d, f"p{q}", "out", "eval_seed0.json")))["rows"]
                        if r["target"] == "pesq4" and r["condition"] == c and r["rule"] == "crc@1")
g = {c: {q: {"success_fa1": row(q, c)["fa1"]["success"], "ber": row(q, c)["ber"]} for q in ("0", p)}
     for c in ("clean", "ota", "ota_opus12")}
# no-go if the codec stage costs more than the sampling noise on clean / ota (96 recordings)
g["go"] = all(g[c][p]["success_fa1"] >= g[c]["0"]["success_fa1"] - 0.10 and g[c][p]["ber"] <= g[c]["0"]["ber"] + 0.03
              for c in ("clean", "ota"))
json.dump(g, open(os.path.join(d, "opus_gate.json"), "w"), indent=1)
print("opus gate", json.dumps(g))
PY
    rm -rf "$RUN_DIR"/out/p*/ckpt
  fi
  python -c "import json,sys; sys.exit(0 if json.load(open('$gate'))['go'] else 3)" || { echo "opus gate: no-go"; exit 3; }
  MODE=opus_train
fi
if [ "$MODE" = opus_train ]; then          # ours trained with the Opus stage (main-run steps / settings), tag _opus
  for s in ${SEEDS:-0 1 2}; do
    python run_a1.py train --seed "$s" --steps "${STEPS:-20000}" --opus_p "${OPUS_P:-0.5}"
    python run_a1.py eval --seed "$s" --methods ours --tag _opus --conditions $OPUS_CONDS
    python run_a1.py summary
  done
  if [ -n "${A1_BASE_CKPT:-}" ]; then      # the main-run models and the baselines on ota_opus12, kept in a separate summary
    for s in ${SEEDS:-0 1 2}; do
      RUN_DIR=$RUN_DIR/out/base A1_CKPT=$A1_BASE_CKPT python run_a1.py eval --seed "$s" --tag _base_opus12 --conditions ota_opus12
    done
    RUN_DIR=$RUN_DIR/out/base python run_a1.py summary
  fi
  exit 0
fi
STEALTH_METHODS="ours ours_k32 ours_k48 dear"
STEALTH_CONDS="clean reverb ota ota_mp3 ota_noisy"
if [ "$MODE" = stealth_tiny ]; then       # smoke test of the stealth / payload evaluation
  python run_a1.py eval --seed 0 --tiny --tag _stealth_tiny --no_native --methods $STEALTH_METHODS     --conditions clean ota --pesq_targets 4.3 4.6
  exit 0
fi
if [ "$MODE" = stealth ]; then            # stealth sweep (PESQ up to 4.6) x payload (16 / 32 / 48 bits per 8 s), models from $A1_CKPT
  for s in ${SEEDS:-0 1 2}; do
    python run_a1.py eval --seed "$s" --tag "${TAG:-_stealth}" --no_native --methods ${METHODS:-$STEALTH_METHODS}       --conditions ${CONDS:-$STEALTH_CONDS} --pesq_targets ${PESQ_TARGETS:-4.0 4.3 4.5 4.6} --n_rec "${N_REC:-300}"
  done
  exit 0
fi
if [ "$MODE" = train_variant ]; then      # ours retrained with other settings ($TRAIN_ARGS), then the stealth sweep on it
  for s in ${SEEDS:-0}; do
    python run_a1.py train --seed "$s" --steps "${STEPS:-20000}" ${TRAIN_ARGS:-}
    python run_a1.py eval --seed "$s" --tag _stealth --no_native --methods ours ours_k32 ours_k48       --conditions ${CONDS:-$STEALTH_CONDS} --pesq_targets ${PESQ_TARGETS:-4.0 4.3 4.5 4.6}
  done
  exit 0
fi
if [ "$MODE" = fair_tiny ]; then          # smoke test: baseline bounds (AudioSeal oracle grid, WavMark soft) + measured rooms
  python run_a1.py eval --seed 0 --tiny --tag _fair_tiny --no_native --methods ours audioseal wavmark wavmark_soft     --conditions clean ota_mit --pesq_targets 4.0
  exit 0
fi
if [ "$MODE" = fair ]; then               # baseline bounds and the held-out measured-room channel, models from $A1_CKPT
  for s in ${SEEDS:-0 1 2}; do
    python run_a1.py eval --seed "$s" --tag "${TAG:-_fair}" --no_native --methods ${METHODS:-ours audioseal wavmark_soft dear}       --conditions ${CONDS:-clean ota ota_mit ota_mit_mp3} --pesq_targets ${PESQ_TARGETS:-4.0 4.3} --n_rec "${N_REC:-300}"
  done
  exit 0
fi
if [ "$MODE" = audioseal_ft ]; then       # AudioSeal fine-tuned on our training channel, then the fair evaluation on it
  for s in ${SEEDS:-0}; do
    python ft_audioseal.py --seed "$s" --steps "${STEPS:-8000}" ${FT_ARGS:-}
    python run_a1.py eval --seed "$s" --tag _audioseal_ft --no_native --methods audioseal_ft       --conditions ${CONDS:-clean ota ota_mp3 ota_mit ota_mit_mp3} --pesq_targets ${PESQ_TARGETS:-4.0 4.3}
  done
  exit 0
fi
if [ "$MODE" = speed ]; then
  python run_a1.py train --seed 99 --steps 400 --batch 32 --ckpt_every 200
  exit 0
fi
SEEDS=${SEEDS:-"0 1 2 3 4"}; STEPS=${STEPS:-40000}
if [ "$MODE" = evalonly ]; then           # standard evaluation of already-trained models (A1_CKPT)
  for s in $SEEDS; do python run_a1.py eval --seed "$s"; done
  python run_a1.py summary
  exit 0
fi
if [ "$MODE" = stress ]; then             # evaluation only, unseen channels; models from A1_CKPT
  for s in $SEEDS; do
    python run_a1.py eval --seed "$s" --tag _stress --conditions ota_noisy speed_1pct phone opus_12k aac_32k
  done
  python run_a1.py summary
  exit 0
fi
if [ "$MODE" = real_make ] || [ "$MODE" = real_score ]; then   # real loudspeaker/microphone check (real_ota.py)
  python real_ota.py "${MODE#real_}" --tag "${REAL_TAG:-pilot}" --n "${REAL_N:-6}" --seed "${REAL_SEED:-0}" --stream_seed "${REAL_STREAM_SEED:-30000}" ${REAL_ARGS:-}
  exit 0
fi
if [ "$MODE" = energy ]; then             # energy-matched comparison: every method at the same host-to-watermark SNR
  for s in $SEEDS; do
    python run_a1.py eval --seed "$s" --tag _energy --no_native --pesq_targets --snr_targets ${SNR_TARGETS:-22 26}       --methods ${METHODS:-ours audioseal wavmark} --conditions ${CONDS:-clean ota ota_mp3 ota_noisy}
  done
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
