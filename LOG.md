# Lab log

Newest entry first.

## 2026-11-24 (session 2): reports written, H2 settled, B7 launched

### Verdicts settled this session
- **A1 (over-the-air watermarking): PASS, report written** (`reports/2026-11-24-a1-ota-watermark`). Built the 5-seed
  simulated table from the corrected seed-0 eval (job supersede) + seeds 1-4 with fixed folds. On both re-recorded
  channels (`ota`, `ota_mp3`) at both matched-PESQ levels, ours clears the nearest baseline by 0.65-0.90 absolute with
  non-overlapping seed spreads (criterion was 0.10). Energy-matched eval (seed 0) confirms the win is not a PESQ
  artifact: AudioSeal fails `ota` (0.05) even when given *more* residual energy than ours. Real loudspeaker/microphone
  full set (seed 0) already measured: ours 1.00/0.94 at PESQ 4.0/4.3, baselines ~0. Stress set (seed 0 preview): ours
  survives phone/aac/speed/noise; Opus-12k after a real room breaks it. Ablation (2 seeds): realistic-channel training
  adds only ~0.01 on in-distribution `ota` — its value must be shown on stress/real (not yet run on the simple-channel
  model). HYPOTHESES A1 -> done/pass.
- **A3 (LLM-generated weights): FAIL, report written** (`reports/2026-11-24-a3-llm-weights`). 7B generator 0.668 vs its
  own fitted scaffold 0.687 (−0.018, CI excludes 0); gated variant silences the generator and lands on the scaffold;
  model size 0.5B/1.5B/7B all the same; TabPFN v2 0.699 leads. The LLM-written weights add nothing over the fitted
  linear read-out, as at CPU scale. HYPOTHESES A3 -> report path updated.
- **B4 (NCA re-seed schedule): H2 settled at 30 seeds** (`reports/2026-11-24-b4-nca-schedule` updated;
  `experiments/b4_nca_schedule/merge_h2.py`, `results/b4_nca_schedule/h2_merged.json`,
  `figures/fig4_h2_30seeds.png`). Pooling seeds 0-29: H1 holds (2->8 beats const2 on anytime 30/30, +0.019). H2 does
  NOT meet the pre-stated 80%-of-seeds bar — 2->8 beats equal-data const5 on only 21/30 (70%) — though the mean gain
  is real (+0.017, CI [+0.008, +0.027] excludes 0). const5 ties const2. A lower-starting wider ramp 1->8 is the best
  arm (ties 2->8, beats const2 18/20); 2->12 overshoots and hurts persistence. HYPOTHESES B4 -> done.

### Launched this session
- **B7 step 1** (`experiments/b7_hard_al`, new): does typicality lose its edge when it is a poor prior? Reuses the A2
  library and cached DINOv2 features; runs the 7 baseline rules on 3 hard pools — long-tailed CIFAR-100 (rho=100),
  CIFAR-10 with 40% symmetric and 40% asymmetric label noise — 10 seeds, no policy. Pre-stated: if some rule beats
  TypiClust by >=0.01 on >=8/10 seeds on any (task, readout), the full B7 policy build is justified; else B7 inherits
  A2's negative. Tiny validated (job done). Full run submitted (small tier). Tiny preview (not a result): on
  c100_longtail, Coreset/BADGE/Margin beat TypiClust by ~0.08 — the direction looks right.

### Reviewed, not pursued
- Newest Random-Ideas daily `2026-11-24-trig-fanchirp-poles` (per-partial de Moivre via ESPRIT poles) is a CPU DSP
  audio-effects study, ~4.8 s/s of audio on numpy, not GPU-bound — outside the lab's strong-GPU mission. No new
  GPU-worthy hypothesis from it.

### Handoff for the next session
- **Running (all within limits): strong = 2** (1288962 A1 ablation finishing seeds 0-2; 1294033 DeAR baseline training,
  ~14 h); **small = 3** (1293870 A1 stress seeds 0-4; 1294034 A1 energy seeds 0-4; 1294221 B7 full baselines, 8 h).
- **Check next:**
  1. **B7** (1294221): read `results/b7_hard_al/.../out/summary.json` -> `headroom` list. If non-empty on a real
     (non-tiny) run, write `reports/<date>-b7-hard-al` and decide whether to build the noise/imbalance-aware policy
     (strong-GPU). If empty, record B7 as inheriting A2's negative. Do a quick web check on class-imbalanced /
     noisy-label AL before investing in the policy.
  2. **A1**: when stress (1293870) and energy (1294034) finish all 5 seeds, refresh the A1 report's stress and
     energy tables from seed-0-only to 5-seed (rebuild via the clean-5-seed assembly in this session's notes). When
     DeAR (1294033) finishes, add it as a baseline row. Run the simple-channel ablation on the stress + real sets to
     justify the realistic channel.
  3. **Strong slot** frees when 1288962 ends with nothing queued. Candidates to fill it, highest-value first: B1
     (Forward-Forward: is the loss the problem, not locality? — needs CIFAR/ViT training, genuinely strong-GPU) or B3
     (MoE vs dense at GPT-2 scale). A1 32-bit payload needs a link-layer code change (not an env var).
- **Known problems:** A1 stress/energy are seed-0-only in the current report tables (jobs still running). A1 WavMark
  decoder is slow (~2.5 h/seed). The A1 report's energy and stress sections are explicitly marked preliminary.
- **Confidentiality:** grep clean on all pushed files (the only matches are HuggingFace dataset shard names like
  `train-00000-of-...` matching the node-name regex incidentally — not host names).

## 2026-11-24, results so far and handoff

### Verdicts
- **A2 (learned active-learning policy): fail.** TypiClust is the best rule in all 12 cells on frozen DINOv2 features;
  the RL policy loses on 0/10 seeds (-0.045 to -0.069 AUC); imitation of TypiClust ties it and RL fine-tuning only
  hurts. Report: `reports/2026-11-24-a2-al-policy`.
- **B4 (NCA re-seed schedule): rising 2 -> 8 beats constant 2** on anytime accuracy on 10/10 seeds (+0.018); whether it
  is the timing or just the amount of fresh data is open (6/10 against constant 5). Report:
  `reports/2026-11-24-b4-nca-schedule`. Follow-up with 20 more seeds is running.
- **A3 (LLM-generated weights with a real LLM): fail.** Mean accuracy over 23 held-out OpenML tables, k = 1..20:
  7B generator 0.668 +- 0.002 (3 seeds) against its own fitted scaffold 0.687; 1.5B and 0.5B generators 0.673 against
  0.686. The gated variant only ties the scaffold (0.687). References: TabPFN v2 0.699, untrained ridge scaffold 0.685,
  logistic regression 0.681, 7B in-context prompting 0.589. The written weights add nothing over the linear read-out,
  as in the CPU-scale runs. Report still to be written.

### Interim (not verdicts)
- **A1, seed 0 of 5** (300 recordings of 8 s, 16-bit messages, random start, message success at <= 1 % false alarm):

  | channel | ours, PESQ 4.0 | ours, PESQ 4.3 | AudioSeal, PESQ 4.0 | AudioSeal, PESQ 4.3 | WavMark |
  |---|---|---|---|---|---|
  | clean | 1.000 | 1.000 | 0.69 | 0.63 | 1.000 |
  | MP3 64k | 1.000 | 1.000 | 0.66 | 0.61 | 0.990 |
  | reverb (held-out rooms) | 1.000 | 0.990 | 0.03 | 0.04 | 0.500 |
  | over the air (`ota`) | 1.000 | 0.987 | 0.04 | 0.04 | 0.263 |
  | over the air + MP3 | 0.983 | 0.960 | 0.03 | 0.02 | 0.150 |

  The WavMark column is at its native strength (PESQ 4.17); its matched-PESQ rows were still running.

  Ours uses CRC-8 with no threshold (raw false alarm 0-0.3 %). At native strength our watermark sits at PESQ 3.83
  (SI-SNR 20 dB); WavMark's native PESQ is 4.17 and AudioSeal's 4.47. AudioSeal numbers in this table come from the
  first evaluation, whose threshold folds were split by source type (realised false alarm 7-11 %, so they are if
  anything optimistic for AudioSeal); a corrected rerun of seed 0 is queued. The test channels come from the same
  simulator family as training, so the stress set of unseen channels and real recordings are needed before any claim.
- **A1, seeds 0-2 with the corrected folds** (seed 0 from the eval-only rerun; WavMark only 1 seed so far), message
  success at <= 1 % false alarm on `ota` / `ota_mp3`: ours 0.998 / 0.993 at PESQ 4.0 and 0.981 / 0.976 at PESQ 4.3;
  AudioSeal 0.037 / 0.038 and 0.052 / 0.032; WavMark 0.313 / 0.243 and 0.103 / 0.083. Seeds 3-4 are training.
  (The seeds 0-2 job is marked failed only because its shell script was updated under it after the last step; all
  three seeds' training and evaluation files are complete.)
- **A1 real loudspeaker / microphone pilot** (laptop built-in speakers and microphone array, one room, 6 host streams
  per arm, 10 min): every item aligned (correlation median 0.41, min 0.26; recording level -49 dB RMS, no clipping).
  Ours decodes 6/6 at PESQ 4.0 and 4.3 with CRC and no false alarm on the 6 unmarked twins; AudioSeal 0/6 (bit
  error 0.24-0.29) and WavMark 0/6 (bit error 0.56) at both strengths. Too small for a claim; the 2.8 h full set
  (100 streams per arm) is recording. Results: `results/a1_ota_watermark/1293103/real_pilot.json`.
- **A1 real loudspeaker / microphone, full set** (same laptop and room, 100 host streams per arm, 50 speech / 50
  music, 2.8 h, 1000 items; alignment correlation median 0.40; level -55 dB RMS, quieter than the pilot, no
  clipping). Message success at <= 1 % false alarm (two-fold thresholds), n = 100 per cell:

  | method | PESQ 4.0 | PESQ 4.3 |
  |---|---|---|
  | ours (CRC, list of 4) | 1.00 | 0.94 |
  | AudioSeal | 0.00 | 0.02 |
  | WavMark | 0.00 | 0.00 |

  Ours: no CRC false alarm on the 100 unmarked twins; bit error 0.000 at PESQ 4.0, 0.032 at 4.3. AudioSeal still
  *detects* its mark (AUC 0.999 / 0.974) but cannot read the message (bit error 0.31); WavMark finds nothing (AUC 0.50,
  bit error 0.50). One room, one device pair, one distance, ours seed 0 only. Results:
  `results/a1_ota_watermark/1293800/real_full.json`.
- **A1 training-channel ablation** (ours trained on the original simple channel only, 3 seeds, simulated test
  channels, message success at <= 1 % false alarm): `ota` 0.982 +- 0.010 / 0.853 +- 0.067 at PESQ 4.0 / 4.3 (full
  channel: 0.998 / 0.981), `ota_mp3` 0.970 / 0.817 (full: 0.993 / 0.976), held-out reverb 0.996 / 0.943. The
  realistic loudspeaker / room / microphone training channel matters mostly at the quieter strength.
- **A1 Opus in the training channel** (queued): real-room + Opus 12k was the one condition that broke ours, so an
  optional straight-through Opus stage (8-24 kbit/s, p = 0.5) was added to training, with a new `ota_opus12` test
  condition for all methods. A 4000-step go/no-go check (no loss on clean / `ota`) runs before 3 x 20k-step training.
- **A1 real set 2: baselines at matched energy** (same laptop and room, 2.8 h, 100 streams per arm; ours seed 1 at
  PESQ 4.0 / 4.3; AudioSeal and WavMark scaled to host-to-watermark SNR 22 and 26 dB, i.e. the energy ours carries at
  PESQ 4.0 / about 4.3; WavMark is +12.7 dB louder than at matched PESQ). Message success at <= 1 % false alarm:

  | method | PESQ 4.0 / SNR 22 | PESQ 4.3 / SNR 26 |
  |---|---|---|
  | ours, seed 1 (CRC, list of 4) | 1.00 | 0.85 |
  | AudioSeal | 0.00 (bit error 0.32) | 0.00 (0.40) |
  | WavMark | 0.00 (0.48) | 0.00 (0.49) |

  Giving the baselines our energy does not help them in the room, so the gap is not just a loudness difference.
  Ours seed 1 shows a raw CRC false alarm of 5 % on the unmarked twins (seed 0: 0 %); the table uses the thresholded
  score, which holds false alarm at <= 1 %. Results: `results/a1_ota_watermark/1294317/real_set2.json`.
- **A1 real recording, then a codec** (the same 2.8 h full-set recording, re-encoded after the room; CRC, list of 4,
  message success at <= 1 % false alarm, n = 100 per cell):

  | after the room | ours, PESQ 4.0 | ours, PESQ 4.3 | AudioSeal 4.0 / 4.3 | WavMark 4.0 / 4.3 |
  |---|---|---|---|---|
  | none | 1.00 | 0.94 | 0.00 / 0.02 | 0 / 0 |
  | MP3 64k | 0.98 | 0.82 | 0.00 / 0.00 | 0 / 0 |
  | AAC 32k | 0.99 | 0.82 | 0.01 / 0.02 | 0 / 0 |
  | telephone band (300-3400 Hz + mu-law) | 0.97 | 0.47 | 0.03 / 0.01 | 0 / 0 |
  | Opus 12k | 0.40 | 0.04 | 0.00 / 0.00 | 0 / 0 |

  Ours keeps working through MP3 / AAC after a real room and mostly through a phone band at PESQ 4.0; low-rate Opus
  (12 kbit/s, never seen in training) breaks it. CRC false alarm stays at 0-2 % on the unmarked twins. The baselines
  were already at chance before the codec. Results: `results/a1_ota_watermark/1293888/real_full_post.json`.
- **A1 audibility caveat and a first blind listening check.** Matching is on speech PESQ only, and at PESQ 4.0 ours
  carries about 11 dB more residual energy than WavMark (SI-SNR 22 vs 33.5 dB; AudioSeal 18 dB), so a PESQ blind spot
  is possible (the mask is host-shaped, i.e. EQ-like, which PESQ partly compensates). One listener, headphones not
  controlled, 10 blind sets (5 speech, 5 music), each the same clip with the three marks at PESQ 4.0 in shuffled
  order, "which sounds best": WavMark 4, ours 2, AudioSeal 1, no difference 3. No method stands out as clearly
  worse; far too small to separate them. Still needed: ABX against the unmarked original, a second metric (ViSQOL
  audio mode), and the baselines at matched energy.
- **A1 prior art (literature check).** Learned over-the-air watermarking is not new: DeAR (AAAI 2023, arXiv
  2212.02339) trains an encoder/decoder through a re-recording distortion layer and reports about 98.5 % bit accuracy
  at 20 cm on real re-recordings (SNR 25.9 dB). The SoK of arXiv 2503.19176 already reports that current watermarks,
  AudioSeal included, fail physical-level distortion, so "AudioSeal/WavMark break over the air" is not new either. What
  A1 adds: blind decoding (unknown timing and phase) with message-level success at a fixed false-alarm rate, a
  matched-audibility comparison with the modern baselines, and a real-room test. For a paper it needs DeAR as a
  baseline, energy-matched baselines, several rooms / distances / devices, and a formal listening test.

### Handoff for the next session
- **Running / queued:**
  - A1 main run: seeds 0-2 (training + evaluation, about 3.5 h per seed), then seeds 3-4 in a chained job, then the
    training-channel ablation (3 seeds, ours only), all on one 80 GB GPU in sequence.
  - A1 corrected re-evaluation of seed 0: done. The stress-set evaluation of all 5 seeds is queued (all seeds trained;
    the seed 0-2 and 3-4 jobs show as failed only because run.sh was updated under them after their last step).
  - B4 follow-up: seeds 10-29 of constant 2, constant 5, 2 -> 8 and the ramp end-points 2 -> 12, 1 -> 8 (two jobs,
    finished; results not merged yet).
  - A1 real loudspeaker / microphone check (`real_ota.py`, `real_play.py`): playlists built for a 10 min pilot
    (6 streams per arm) and a 2.8 h full set (100 streams per arm; ours seed 0, AudioSeal, WavMark at PESQ 4.0 and 4.3,
    plus unmarked twins, shuffled). Playback on a laptop's built-in speakers and microphone array:
    pilot and full set done (results above).
  - A1 DeAR baseline (`dear.py`, reimplemented from the paper; no public code): 3 seeds x 250k steps training on
    one 80 GB GPU, then the standard evaluation with `--tag _dear` in the same job. Deviations from the paper are in
    the module docstring. Training stalled at chance with the paper's loss weights from step 0, so the fidelity terms
    are warm-started (after that the objective is the paper's); body convolutions are dilated. The blind segment sync
    has not yet been checked against the true offsets. 4000-step tiny test: training bit error 0.20 and still falling
    (not a result).
  - A1 energy-matched simulated evaluation (seeds 0-2; AudioSeal / WavMark at host-to-watermark SNR 22 and 26 dB)
    and real set 2 (ours seed 1, baselines at matched SNR) are running.
- **Check next:**
  1. A1: use the corrected seed-0 rerun and seeds 1-4 (fixed folds; per-recording scores saved). Build the table with
     `python experiments/a1_ota_watermark/make_figures.py results/a1_ota_watermark <out>`; apply the pre-stated
     criterion on `ota` / `ota_mp3`; read the stress set and ablation; write `reports/<date>-a1-ota-watermark`.
  2. A3: write the report from `results/a3_llm_weights/1288777/summary.json` (verdict above).
  3. B4: merge seeds 0-29 for the H2 test (constant 5 vs 2 -> 8) and the ramp end-points.
- **Known problems:**
  - A1's WavMark decoder is slow (about 0.65 s per 8 s recording); each seed's evaluation takes about 2.5 h.
  - AudioSeal's clean success stays below 1 even with the grid search (music is weaker: 0.4-0.6 vs 0.7-0.8 speech).
    Check whether a finer search or decoding on 1 s segments with voting helps it before the final comparison.
  - The A1 in-distribution channels are near ceiling for ours; the stress set and a real loudspeaker / microphone
    recording set are the real test.
  - A2: CIFAR-10N labels were not downloaded; ProbCover is approximate.

## 2026-11-24 (first strong-GPU session)

Launched the four experiments that the daily runs could not do on their hardware. Each was validated at tiny
scale before its full run.

- **A1, over-the-air watermarking vs AudioSeal / WavMark** (`experiments/a1_ota_watermark`). New physical layer
  (about 1 M parameter encoder, 2 M decoder), trained on 15 h LibriSpeech + 8.8 h FMA music through a simulated
  loudspeaker -> room (image-source RIRs, disjoint test rooms) -> microphone chain with real background sounds;
  the original link layer (conv code, CRC-8, list Viterbi, joint timing search) is reused unchanged. Baselines are the
  public AudioSeal 16-bit and WavMark checkpoints at matched PESQ (4.0, 4.3) and at native strength. Full run:
  5 seeds x 20k steps, split into two chained jobs on one 48 GB GPU. Pass/fail is in the README.
  - Finding while validating: the public AudioSeal 16-bit decoder only recovers the message when the recording is
    aligned to a 320-sample grid (a 10-sample shift drops it to chance), although detection itself is
    shift-tolerant. Our benchmark gives AudioSeal a grid search (4-sample steps + 1-sample refinement on a 2 s probe)
    so the comparison is not decided by this. WavMark's own pattern search is shift-robust (unchanged at shifts of
    7 to 5000 samples).
  - Throughput: 0.62 s per step at batch 32 three-chunk streams after loading the corpora into RAM (memory-mapped
    random reads were the first bottleneck).
- **A3, LLM-generated weights with a real LLM** (`experiments/a3_llm_weights`). A pretrained Qwen2.5 LLM (LoRA) reads
  k labelled rows per class plus column names and writes deltas on a d-128-C MLP whose second layer is a ridge
  read-out refitted per episode (the fitted scaffold). Meta-trained on SCM-prior synthetic tasks only; evaluated on
  23 held-out OpenML tables, 2300 frozen episodes, k = 1..20. Baselines: silenced scaffold, logistic regression,
  TabPFN v2, k-NN, and in-context prompting of the same 7B LLM. Arms: 7B x 3 seeds, 1.5B x 3, 0.5B x 3, a
  random-init 0.5B control. Full run on one 80 GB GPU. PASS = 7B generator beats both its scaffold and 7B in-context
  prompting (95 % bootstrap CI over tables above 0).
- **A2, learned active-learning policy** (`experiments/a2_al_policy`). Frozen DINOv2 ViT-B/14 features for
  CIFAR-10/100; Random, Entropy, Margin, Coreset, TypiClust, ProbCover and BADGE with a linear probe and label
  propagation, budgets 10-1000, 10 seeds; a group-relative-RL MLP policy meta-trained on 10-way CIFAR-100 tasks
  (classes 0-49) and tested on CIFAR-10, held-out CIFAR-100 classes 50-99 and full CIFAR-100. Running on 11 GB GPUs.
- **B4, NCA re-seed schedule** (`experiments/b4_nca_schedule`). 6 schedules x 10 seeds of the persistent
  self-classifying NCA (constant 2/5/8, linear 8->2, linear 2->8, step 2->8 at 50 %), with the original
  evaluation protocol. Running on 11 GB GPUs.

## 2026-11-23
- Repository created.
- Backlog seeded in `HYPOTHESES.md` from the Random-Ideas daily experiments.
