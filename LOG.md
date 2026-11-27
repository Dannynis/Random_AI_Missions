# Lab log

Newest entry first.

## 2026-11-26: B7 settled (PASS, replicated at large backbone), B1 launched at scale

Start state: **no lab jobs running or queued** — the 11-24/11-25 A1 real-room sweep and the stress/energy/DeAR/B7
jobs had all finished (the A1 interim handoff below was kept current through ~09:13 today). Full budget free. Owner's
`awesome` job left alone. The big A1 exploration is effectively complete and needs the owner's hardware
for more real rooms, so this session moved the lab's compute onto the open B-tier questions.

### Verdicts settled this session
- **B7 step 1: PASS, report written** (`reports/2026-11-26-b7-hard-al`). Full run 1294221 (10 seeds × 7 rules × 3 hard
  tasks × 2 readouts) scored against the pre-stated bar (a rule beats TypiClust by ≥0.01 AUC on ≥8/10 seeds on any
  (task, readout)). Cleared: on long-tailed CIFAR-100 **Coreset beats TypiClust +0.063 AUC (lp, 10/10)** / +0.028
  (linear); on 40% symmetric noise **Margin/BADGE beat it +0.045/+0.034 (linear, 8/10)**. At ViT-B/14, asymmetric
  noise had no headroom. So typicality is a poor prior exactly where coverage (long-tail) or uncertainty (noise) is the
  right geometry. HYPOTHESES B7 → running (policy build justified).
- **B7 replicated and widened at dinov2-large (ViT-L/14)** (job 1303503, small tier, report section added). Same 210
  runs on ViT-L/14 features: Coreset long-tail +0.067 lp (10/10), Margin symmetric +0.060 (8/10), and **Margin now
  also clears the bar on asymmetric noise +0.059 linear (8/10)** — the one regime that looked rule-less at the smaller
  backbone. The verdict is not a backbone artifact; "coverage wins the tail, uncertainty wins noise" holds at scale.
  Prior-work check logged: the policy's real bar is DIRECT (arXiv 2312.09196) and the ICML-2026 foundation-prior AL
  method, not just the seven generic rules.

### Launched this session
- **B1 (new): layer-local vs backprop depth sweep on CIFAR** (`experiments/b1_ff_depth`). Lifts the MNIST
  Forward-Forward decomposition (Random-Ideas #46 supervised, #60 self-supervised — both found locality is free, the
  threshold loss is the cost, but depth adds nothing for a local learner on MNIST) to CIFAR, where features must
  compose across depth. Three rules at matched conv arch/epochs: `bp` (end-to-end), `local_ce` (greedy layer-local
  cross-entropy), `ff` (greedy layer-local goodness, pairwise SymBa, BN disabled). Depth {4,8,12} × CIFAR-10/100 ×
  3 seeds = 54 cells, resumable with epoch-level checkpoints, 3 workers share one GPU. Pre-stated H1 (best local within
  3 pp of bp on CIFAR-10) and H2 (deepest-block probe beats first by ≥5 pp — does locality compose with depth on a hard
  task?). Smoke-tested all rules + preempt/resume locally; tiny validated on cluster (1303585, 12 cells, DONE). Full
  run **submitted on a single strong GPU, 24 h, resumable** (1303586, pending at log time).
- **B7-large** (1303503, small tier) — the backbone-robustness re-run above. Done.

### Reviewed, not pursued
- Random-Ideas `2026-11-26-horn-channel-calib`: CPU DSP audio-matching (segmented GCC-PHAT), not GPU-bound — outside
  the strong-GPU mission. `2026-11-25-ff-selfsup-pairwise` IS pursued — it is the direct motivation for B1.

### Infra notes
- CIFAR downloads from the public torchvision mirror are slow (~100 kB/s) and the first tiny job stalled with 3 workers
  contending on one partial file. Fixed B1's `run.sh` to **prefetch both datasets once (serially) before spawning
  workers**; pre-staged CIFAR-10/100 into the shared data cache (one-time ~50 min). Added `dinov2-large` to the A2
  extractor (one line) for the B7 replication. Guarded the SIGUSR1 handler so the code imports on Windows too.

### Handoff for the next session
- **Running/queued:** strong = 1 (**B1 full 1303586**, pending→running on a strong GPU, 24 h, resumable; collect with
  `python experiments/b1_ff_depth/b1_depth.py collect` after sync). small = 0. Budget free: 1 strong + 3 small.
- **Check next:**
  1. **B1** (1303586): `bash infra/sync_results.sh`, read `results/b1_ff_depth/1303586/out/summary.json`. Apply H1
     (per (dataset,depth): best of local_ce/ff `head_acc_mean` vs bp within 3 pp on CIFAR-10) and H2 (`depth_gain_mean`
     ≥0.05 for the best local rule) — compare the local depth curve to bp's. Write `reports/<date>-b1-ff-depth`. If the
     run is slow, it chains within 24 h via per-cell ckpt; resubmit the same command to resume if preempted.
  2. **Fill the idle budget** with the two ready builds: **B3 (MoE vs dense at GPT-2 scale)** on a strong GPU — the
     next flagship, genuinely LLM-scale; and the **B7 noise/imbalance-aware policy** on the small tier — now justified
     and backbone-robust, must be benchmarked against DIRECT and the foundation-prior AL method (not just the generic
     rules), per-regime best fixed rule (Coreset/Margin) as the imitation ceiling.
- **Known problems:** torchvision CIFAR mirror is slow — keep datasets in the shared cache (now staged), never let many
  workers download concurrently. B1's `ff` arm may underperform on CIFAR (goodness on conv is hard); a weak `ff` is a
  reportable negative and does not affect the `local_ce`-vs-`bp` headline or the depth-composition curve.
- **Confidentiality:** grep clean on all pushed files this session (only incidental matches are dataset names like
  `train-clean-100` / `ImageNet-100` hitting the node regex, not host names).

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
- **A1 DeAR baseline: first evaluation discarded, decoder input fixed.** The first DeAR evaluation (3 seeds, 250k
  steps) gave near-zero message success everywhere, even clean at native strength. Checks on 32 clean streams: the
  blind segment sync is not the problem (true-offset decoding gives the same bit error, 0.115 vs 0.123); BatchNorm
  statistics are not either (re-estimating them changes nothing). The decoder was only ever trained on audio after the
  distortion layer's 1-4 kHz band filter, and applying that fixed filter before decoding brings clean bit error at
  native strength from 0.115 to 0.008 (message 0.97) and at PESQ 4.0 from 0.30 to 0.11 (message 0.31). The filter is
  now part of DeAR's decoder. DeAR's native strength is speech PESQ 2.3; reaching PESQ 4.0 takes -11.6 dB (ours:
  -2.1 dB). Re-evaluation, 3 seeds, message success at <= 1 % false alarm:

  | DeAR at | clean | reverb | ota | ota_mp3 |
  |---|---|---|---|---|
  | native (PESQ 2.3) | 0.830 +- 0.196 | 0.689 | 0.478 +- 0.094 | 0.432 |
  | PESQ 4.0 | 0.054 | 0.001 | 0.000 | 0.000 |
  | PESQ 4.3 | 0.000 | 0.000 | 0.000 | 0.000 |

  The reimplementation does carry its message over the simulated air at its own (clearly audible) strength, but at the
  audibility of the comparison (PESQ 4.0 / 4.3) it decodes nothing over the air (ours: 0.998 / 0.981). Caveats:
  reimplementation with guessed details (module docstring), no CRC / detection head (it has to rely on its score to
  hold false alarm). Results: `results/a1_ota_watermark/1295468`.
- **A1 review fixes, part 2: AWARE baseline, room false alarms, room seeds, decision rule.**
  - AWARE (arXiv 2510.17512, code pinned at commit 402041a, imported from its source; 16 info + 4 zero bits of its
    20), 3 seeds x 300, message success at <= 1 % false alarm with Wilson 95 % intervals, PESQ 4.0 / 4.3: clean 0.973
    [0.96, 0.98] / 0.836; simulated air 0.439 [0.41, 0.47] / 0.201; measured rooms 0.699 [0.67, 0.73] / 0.448; measured
    rooms + MP3 0.662 / 0.404. The strongest baseline so far (AudioSeal and WavMark are near 0 over the simulated air),
    but well below ours at equal PESQ. Results: `results/a1_ota_watermark/1305700`, `1305701`, `1305702`.
  - Room false alarms (external speakers at the 6 dB lower level, 150 unmarked clips and 150 per marked cell, both
    mics): with the list of 4 (`crc@4`) the CRC passed on 4/150 (laptop) and 5/150 (phone) unmarked clips, about 3 %,
    above the 1 % claimed; with the CRC on the best candidate only (`crc@1`) 1/150 and 0/150 (1 in 300, 95 % upper bound
    1.9 %), at almost the same success: PESQ 4.0 0.89 [0.83, 0.93] / 0.87 [0.81, 0.92], PESQ 4.3 0.64 / 0.63 (laptop /
    phone, at <= 1 % false alarm). The pre-registered rule for ours is now `crc@1` everywhere (code, tables, figures);
    the earlier room tables above used `crc@4` and their false-alarm claim is withdrawn. This set (new streams, quieter
    playback, phone moved) reads lower than the earlier ones at PESQ 4.3. Results: `results/a1_ota_watermark/1306098`
    (laptop), `1306103` (phone).
  - Room, model seeds 1 and 2 (same setup, 60 per cell, `crc@1`, laptop / phone): seed 1 PESQ 4.0 0.72 / 0.72, 4.3
    0.62 / 0.57; seed 2 1.00 / 1.00, 0.88 / 0.98; no false alarm (0/60). Seed 0 in comparable sets: 0.87-0.98 at 4.0.
    The spread across model seeds in the room is larger than in the simulator (3 seeds within 0.01 there), so real-room
    numbers need several seeds. Results: `results/a1_ota_watermark/1306102`, `1306104` (seed 1), `1306094`, `1306099`
    (seed 2).
  - AWARE in the real room (external speakers, both mics, 40 per cell, new streams): 0.025 at PESQ 4.0 and 0.00 at 4.3
    on both mics; ours in the same playlist 0.88 / 0.90 at 4.0 and 0.55 / 0.80 at 4.3 (laptop / phone, `crc@1`, no
    false alarm). Its simulated-room advantage over AudioSeal and WavMark does not carry over to the real room.
    Results: `results/a1_ota_watermark/1306214` (laptop), `1306216` (phone).
  - AudioSeal fine-tuned on our channel, second attempt (lr 1e-4, 20k steps): again no learning (training bit error
    0.39) and 0 decoding after fine-tuning. Cause found: our channel shifts the audio by up to 1600 samples and the
    public 16-bit decoder only reads its message on a 320-sample grid, so the message loss saw unaligned audio from
    the first step. Third attempt decodes a grid-aligned crop. Results: `results/a1_ota_watermark/1305598`.
- **A1 review fixes, part 1** (code: one pre-registered decision rule per method, `crc@4` for ours and its payload
  variants, `top` for baselines, used by `summary()`, `make_figures.py` and the new `tables.py`, which adds Wilson 95 %
  intervals; AudioSeal oracle-grid and genie-shift upper bounds; WavMark soft combining; a held-out channel `ota_mit` /
  `ota_mit_mp3` with 270 measured rooms from the MIT IR Survey instead of simulated ones; PESQ always the mean over 8 s
  sub-windows; paired decoding windows on the same streams; `audibility.py`; AudioSeal fine-tuned on our channel).
  - Paired listening windows (seed 0, 150 streams of 32 s, the first 8 / 16 / 32 s decoded; PESQ over 8 s sub-windows),
    ours, message success at <= 1 % false alarm, PESQ 4.3 / 4.5 / 4.6: simulated air 8 s 0.993 / 0.780 / 0.120, 16 s
    1.000 / 0.933 / 0.293, 32 s 1.000 / 0.973 / 0.520; measured rooms (`ota_mit`) 8 s 0.987 / 0.860 / 0.187, 16 s
    0.993 / 0.920 / 0.320, 32 s 1.000 / 0.973 / 0.540. The long-window gain holds on identical streams and on measured
    rooms. Results: `results/a1_ota_watermark/1305476`.
  - Audibility beyond PESQ (seed 0, 40 fresh streams, at the PESQ-matched gains; PEAQ-style noise-to-mask ratio without
    an absolute hearing threshold, share of Bark band-frames where the mark exceeds the host's masking threshold),
    speech / music, at PESQ 4.0 / 4.3 / 4.5: ours 4.7 / 4.7 %, 2.9 / 3.0 %, 1.5 / 1.5 %; AudioSeal 14.2 / 11.1 %,
    11.5 / 9.0 %, 7.4 / 5.4 %; WavMark 31.2 / 18.7 %, 24.1 / 12.8 %, 15.7 / 7.3 %; DeAR 23.0 / 15.6 %, 15.3 / 9.5 %,
    8.7 / 4.8 %. Ours' mark follows the host (it is zero where the host is silent), the others add energy in quiet
    passages, where the masking threshold is low; without an absolute threshold this measure may overstate how audible
    very quiet additions are. The SQUIM predicted MOS does not separate the methods (4.52-4.55 for speech, host 4.55),
    so it is not informative here. STOI >= 0.99 for all. A listening-test kit (hosts and marked versions at every
    target for 3 speech and 3 music hosts, with an ABX page) is in the results folder; no listening test has been run.
    Results: `results/a1_ota_watermark/1305485`.
  - AudioSeal fine-tuned on our training channel, first attempt (8k steps, lr 1e-5): did not learn (training bit error
    0.42 -> 0.36) and lost its clean decoding (0.00 after fine-tuning); not a valid control. Retrying at lr 1e-4, 20k
    steps. Results: `results/a1_ota_watermark/1305475`.
  - Smoke test (16 recordings, PESQ 4.0, measured rooms): AudioSeal with the genie shift (best of 80 grid shifts by the
    true message) 0.13 over the air (clean 0.94), so its failure is not a synchronisation artefact; WavMark with soft
    combining 0.75 against 0.63 with its own decoder; ours 0.88. The full 3-seed run is going.
- **A1, playback level test, real room.** Set B replayed on the external speakers 6 dB quieter than its first two
  plays (same laptop mic position; the phone had been moved closer when re-attached), 40 per cell, laptop / phone mic,
  PESQ 4.0 / 4.3 / 4.4 / 4.5: ours 0.98 / 0.90 / 0.73 / 0.40 and 0.98 / 0.98 / 0.95 / 0.68 (first plays at the original
  level, laptop mic: 0.93 / 0.80 / 0.63-0.65 / 0.18-0.28); 4 bit/s at 4.0 0.93 / 0.95, 6 bit/s at 4.0 0.90 / 0.98; DeAR
  native 0.10 / 0.25, 0 at PESQ 4.0 and 4.3. The quieter level did not hurt on the unmoved laptop mic (if anything
  slightly better, possibly less loudspeaker distortion), so the lower 32 s scores of the previous entry are not
  explained by the playback level alone. Results: `results/a1_ota_watermark/1305315` (laptop), `1305316` (phone).
- **A1 review (fairness audit and prior-art check, before any paper).** Fair: same PESQ calibration for all methods on
  separate streams, blind timing for every method, leakage-free two-fold thresholds, one shuffled playlist and global
  gain in the room, disjoint host audio. To fix: (1) the simulated `ota` / `reverb` conditions are drawn from the same
  channel generator and parameter ranges as training (only the rooms are new), so they are in-distribution for ours;
  claims of unseen-channel robustness rest on the stress set and the real rooms; (2) AudioSeal reads 0.69 even clean,
  so an oracle-aligned AudioSeal decode is needed as an upper bound; WavMark should get soft combining across the
  window; (3) the DeAR reimplementation is far weaker than the paper (bit error 0.12 clean at native strength against
  ~1 %), so it is "DeAR-style" until it reproduces a paper number; (4) decision rules differ between tables (`crc@1` in
  the simulated summary, `crc@4` in the room tables) and `summary()` scores `ours_k*` with `top`; (5) audibility rests
  on speech PESQ only; (6) "<= 1 % false alarm" in the room rests on ~20 unmarked items per fold; (7) the capacity
  figure is a genie-timed, condition-calibrated soft-slot information rate, not an achievable blind rate; (8) PESQ over
  32 s windows is not comparable to PESQ over 8 s; (9) no confidence intervals. Closest prior work: DeAR (AAAI 2023,
  real speaker/mic, bit accuracy only), AWARE (arXiv 2510.17512, simulated replay, AudioSeal / WavMark fail), the SoK
  2503.19176 (small real re-recording test), Tai & Mansour (ICASSP 2019, classical over-the-air with sync), DRAW (TIFS
  2024, not read in full). SilentCipher makes no over-the-air claim. Plausibly new: message-level comparison at matched
  audibility on real hardware against several learned methods; learned embedding with blind sync, CRC and list decoding
  end to end over the air; stealth as a function of listening time; measured information rate against delivered rate.
- **A1, is the long-window gain fair to the baselines? Rate at equal bit rate; punctured code on 3 seeds.**
  - Baselines with a 32 s window, simulated (seed 0, 60 recordings, PESQ 4.0, message success at <= 1 % false alarm):
    WavMark clean 1.000, over the air 0.650 (8 s window: 0.31); AudioSeal 0.567 / 0.000; DeAR 0.067 / 0.000. So
    WavMark also gains from listening longer; ours at 32 s holds 0.991 over the air at PESQ 4.5, a stricter
    audibility than the baselines' 4.0. Real room, 32 s window, external speakers (now 6 dB quieter than in the
    earlier sets), phone / laptop mic, 40 per cell: AudioSeal, WavMark and DeAR at PESQ 4.0 all 0.00 (bit error
    0.40-0.51); ours 0.5 bit/s at PESQ 4.5 0.78 / 0.55, 1 bit/s (32-bit message over 32 s) at 4.5 0.63 / 0.35. Ours is
    lower than in the earlier 32 s plays (0.95-1.00 / 0.63-1.00) at the quieter playback level. Results:
    `results/a1_ota_watermark/1304722` (simulated), `1304945` (phone), `1304944` (laptop).
  - Equal bit rate, longer codeword: a 32-bit message over 16 s (2 bit/s) reads 0.977 over the air at PESQ 4.3 and
    0.703 at 4.5, against 0.983 / 0.783 for the 16-bit message over 8 s; a 64-bit message over 16 s (4 bit/s) 0.887 at
    4.3, against 0.923 for 32 bits over 8 s (seed 0). With this code a longer listening window helps only as
    repetition of a short message, not through longer codewords. Results: `results/a1_ota_watermark/1304437`.
  - Punctured code, 3 seeds, over the air, PESQ 4.0 / 4.3 / 4.4: 8 bit/s 0.947 / 0.620 / 0.337, 10 bit/s 0.854 /
    0.433 / 0.179. Results: `results/a1_ota_watermark/1301069`, `1301070`, `1304436`.
- **A1, overnight: longer windows on 3 seeds; a third real setup (the laptop's built-in speakers) with both mics.**
  - Listening window, simulated, 3 seeds (300 recordings for 8 s / 16 s, 150 for 32 s), message success at <= 1 %
    false alarm, PESQ 4.3 / 4.5 / 4.6:

    | window (rate) | over the air | over the air + MP3 | clean |
    |---|---|---|---|
    | 8 s (2 bit/s) | 0.987 / 0.840 / 0.222 | 0.980 / 0.760 / 0.148 | 1.000 / 0.987 / 0.462 |
    | 16 s (1 bit/s) | 0.998 / 0.939 / 0.424 | 0.996 / 0.896 / 0.291 | 1.000 / 0.996 / 0.806 |
    | 32 s (0.5 bit/s) | 1.000 / 0.991 / 0.673 | 1.000 / 0.969 / 0.480 | 1.000 / 1.000 / 0.951 |

    Results: `results/a1_ota_watermark/1301155`, `1303628`, `1303629` (16 s), `1301156`, `1301454`, `1303627` (32 s).
  - Real room, external speakers, played 6 dB quieter than before from minute 69 of the 32 s replay on (owner's
    request), 40 per cell, phone / laptop mic: 32 s window, PESQ 4.5 1.00 / 1.00, PESQ 4.6 0.50 / 0.28 (first play:
    0.95 / 0.63 and 0.28 / 0.00); 16 s window, PESQ 4.4 0.97 / 0.95, 4.5 0.82 / 0.70, 4.6 0.15 / 0.00, and 2 bit/s
    over 16 s at PESQ 4.3 0.95 / 0.93 (first play: 0.93 / 0.78, 0.78 / 0.40, 0.13 / 0.00, 0.90 / 0.58). The phone was
    re-attached before this replay and recorded 10-12 dB louder than before (closer to the speakers). Results:
    `results/a1_ota_watermark/1304452`, `1304459` (32 s), `1304479`, `1304496` (16 s).
  - Real room, the laptop's built-in speakers at 6 dB below their earlier level, same phone and laptop mic, the same
    playlists as before (phone / laptop mic, 40 per cell):
    - set A: ours 0.97 / 0.97 at PESQ 4.0, 0.93 / 0.65 at 4.3; AudioSeal and WavMark 0.00-0.03.
    - set B: ours 1.00 / 0.95, 0.93 / 0.78, 0.85 / 0.48, 0.45 / 0.15 at PESQ 4.0 / 4.3 / 4.4 / 4.5; 4 bit/s 0.95 / 0.80 at
      4.0; 6 bit/s 0.93 / 0.65 at 4.0; DeAR 0.08 / 0.00 at native, 0 at 4.0 and 4.3.
    - set C: 8 bit/s at 4.0 0.62 / 0.20, 10 bit/s at 4.0 0.45 / 0.03, 2 bit/s at 4.4 0.60 / 0.25 (this playlist was also
      the weakest on the external speakers).
    - 32 s window: PESQ 4.5 0.97 / 0.77, PESQ 4.6 0.28 / 0.07.
    Results: `results/a1_ota_watermark/1304450`, `1304495` (set A), `1304478`, `1304487` (set B), `1304460`, `1304486`
    (set C), `1304451`, `1304505` (32 s).
  - Summary of the real room over three speaker setups and two microphones: at PESQ 4.0 ours carries 2 bit/s at
    0.93-1.00 and 6 bit/s at 0.65-0.93; with a 32 s window it carries 0.5 bit/s at PESQ 4.5 at 0.77-1.00. AudioSeal,
    WavMark and DeAR read nothing at PESQ >= 4.0 in any setup.
- **A1, listening longer for more stealth** (`A1_REC_S`: the receiver window is 16 s or 32 s instead of 8 s, so the
  same 16-bit message, repeated every 8 s, is combined over 2 or 4 cycles: 1 or 0.5 bit/s instead of 2; same model, PESQ
  measured over the whole window). Simulated, seed 0, message success at <= 1 % false alarm, over the air:

  | window | PESQ 4.3 | 4.5 | 4.6 |
  |---|---|---|---|
  | 8 s (2 bit/s) | 0.983 | 0.783 | 0.170 |
  | 16 s (1 bit/s) | 0.997 | 0.913 | 0.293 |
  | 32 s (0.5 bit/s, 150 recordings) | 1.000 | 0.973 | 0.520 |

  Over the air + MP3, 32 s: 1.000 / 0.987 / 0.533; clean, 32 s: 1.000 / 1.000 / 0.907. The 32 s job ran out of GPU
  memory in its 4 bit/s part, after the rows above. Real room, 16 s window (jack speakers, 40 per cell, phone / laptop
  mic): PESQ 4.4 0.93 / 0.78 (8 s window, sets B and C: 0.80-0.88 / 0.33-0.65), PESQ 4.5 0.78 / 0.40 (8 s:
  0.70-0.73 / 0.18-0.28), PESQ 4.6 0.13 / 0.00; 2 bit/s over 16 s (32-bit message) at PESQ 4.3 0.90 / 0.58. So the
  stealth limit moves with listening time, about +0.1 PESQ per doubling of the window in the simulator. Results:
  `results/a1_ota_watermark/1301155` (16 s), `1301156` (32 s), `1301382`, `1301383` (room, 16 s).
  Real room, 32 s window (same setup, 40 per cell, phone / laptop mic): PESQ 4.5 0.95 / 0.63, PESQ 4.6 0.28 / 0.00.
  A loud event near the phone around minute 65 of 77 (one minute 35 dB above the rest) is inside this recording.
  Results: `results/a1_ota_watermark/1301452`, `1301453`.
- **A1, more payload with a punctured code; retraining for stealth; room set C.**
  - Punctured convolutional code (`make_layout`): the same model and 128 slots per 8 s, the rate-1/2 code punctured to
    rate 2/3 for 64 info bits (8 bit/s, `ours_k64`) and to 3/4 for 80 bits (10 bit/s, `ours_k80`); CRC-8, list of 4.
    Simulated, 2 seeds x 300, message success at <= 1 % false alarm, PESQ 4.0 / 4.3 / 4.4: over the air 8 bit/s
    0.955 / 0.633 / 0.347, 10 bit/s 0.858 / 0.428 / 0.177; clean 8 bit/s 1.000 / 0.977 / 0.868. With the 2-6 bit/s
    rows above: at PESQ 4.0 the simulated air carries 8 bit/s at 0.95 (measured information rate there 13.6 bit/s).
    Results: `results/a1_ota_watermark/1301069`, `1301070`.
  - Retraining for stealth (seed 0, sweep as above, compared with the original seed 0): (a) trained from scratch at a
    6 dB lower watermark level: did not learn (bit error 0.50 after 20k steps), discarded; (b) the original model
    fine-tuned 8k steps at that level: same success at equal PESQ (over the air 0.977 vs 0.983 at 4.3, 0.787 vs 0.783
    at 4.5), with 5 dB less watermark energy at equal PESQ (SI-SNR 31.1 vs 25.8 dB at 4.3); (c) trained with a 3x
    weight and half the target on the log-spectral-distance loss: LSD 0.37 vs 0.58 dB at PESQ 4.3, success slightly
    lower (over the air 0.957 vs 0.983 at 4.3). So retraining moves the mark to less energy or a flatter spectral
    footprint at the same PESQ, but does not buy robustness at a given PESQ; the scaled-down original model stays the
    best trade-off. Results: `results/a1_ota_watermark/1300907` (b), `1300294` (c).
  - Room set B replayed (same playlist, same placement): phone mic 0.98 / 0.90 / 0.88 / 0.70 at PESQ 4.0 / 4.3 / 4.4 /
    4.5 (first play 0.98 / 0.90 / 0.88 / 0.73), laptop mic 0.93 / 0.80 / 0.63 / 0.18 (0.93 / 0.80 / 0.65 / 0.28); 4 bit/s
    at 4.0 0.95 / 0.85 (phone / laptop), 6 bit/s at 4.0 0.93 / 0.80; DeAR 0.15 / 0.08 at native, 0 at PESQ 4.0 and 4.3.
    Results: `results/a1_ota_watermark/1301067`, `1301068`.
  - Room set C (same setup, new streams, 40 per cell; phone / laptop mic): 8 bit/s at PESQ 4.0 0.83 / 0.23, at 4.3
    0.20 / 0.00; 10 bit/s at 4.0 0.48 / 0.08; 2 bit/s at 4.4 0.80 / 0.33; 4 bit/s at 4.4 0.33 / 0.03; 6 bit/s at 4.3
    0.58 / 0.08. The recording level on the phone was 4 dB lower than in set B (-56 vs -52 dB RMS) and the 2 bit/s PESQ
    4.4 cell, also in set B, fell from 0.88 to 0.80, so this run's channel was somewhat worse. Measured information rate
    on the phone mic at PESQ 4.0: 10.5 bit/s. Best real-room operating points so far on the phone mic: 2 bit/s at
    PESQ 4.4 (0.80-0.88), 6 bit/s at PESQ 4.0 (0.93), 8 bit/s at PESQ 4.0 (0.83). Results:
    `results/a1_ota_watermark/1301117` (phone), `1301118` (laptop).
- **A1 stress set, 5 seeds** (channels never used in training; message success at <= 1 % false alarm, ours / best
  baseline, PESQ 4.0 | 4.3): AAC 32k 1.00 / 0.99 (WavMark) | 1.00 / 0.92; Opus 12k 0.98 / 0.61 (AudioSeal) | 0.68 / 0.56;
  telephone band 1.00 / 0.64 | 0.96 / 0.59; noisy room 0.94 / 0.21 | 0.83 / 0.07; 1 % speed change 1.00 / 0.00 |
  1.00 / 0.00. Results: `results/a1_ota_watermark/1293870`.
- **A1 stealth x payload sweep, simulated** (3 seeds x 300 recordings; same trained models, the mark only scaled down;
  `ours_k32` / `ours_k48` carry 32 / 48 info bits instead of 16 in the same 8 s cycle and 128 slots, i.e. 4 / 6
  instead of 2 bit/s, CRC-8, list of 4). Message success at <= 1 % false alarm, over the air (`ota`):

  | | PESQ 4.0 | 4.3 | 4.4 | 4.5 | 4.6 |
  |---|---|---|---|---|---|
  | ours, 2 bit/s | 1.000 | 0.987 | 0.967 | 0.840 | 0.222 |
  | ours, 4 bit/s | 0.998 | 0.936 | 0.807 | 0.439 | 0.028 |
  | ours, 6 bit/s | 0.992 | 0.866 | 0.672 | 0.286 | 0.006 |
  | DeAR | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
  | SI-SNR of the mark (ours) | 22.5 dB | 26.1 | 27.8 | 30.2 | 35.1 |

  Clean, 2 bit/s: 1.000 up to PESQ 4.4, 0.987 at 4.5, 0.46 at 4.6. Over the air + MP3, 2 bit/s: 0.999 / 0.980 /
  0.946 / 0.760 / 0.148. DeAR needs -10.8 dB (PESQ 4.0) to -25.9 dB (4.6) of gain, ours -2.3 to -15.0 dB.
  Physical-layer information rate (soft-decision rate of the 16 slots/s at the true timing, LLRs refit per condition,
  two folds), over the air: 13.7 bit/s at PESQ 4.0, 10.6 at 4.3, 8.5 at 4.4, 5.7 at 4.5, 1.9 at 4.6 (slot bit error
  0.04 / 0.10 / 0.14 / 0.21 / 0.33). The rate-1/2 convolutional code with CRC uses 2-6 bit/s of that, so the
  remaining limit on payload is the code, not the channel: at PESQ 4.0 a stronger code could carry several times more.
  Results: `results/a1_ota_watermark/1300133`, `1300134`, `1300181`.
- **A1 real room, second setup: external speakers on the audio jack**, same laptop and room, two microphones (the
  laptop's own array, and a phone's microphone captured over USB, placed elsewhere in the room). 40 host streams per
  cell; message success at <= 1 % false alarm (two-fold thresholds), ours with CRC and a list of 4:

  | playlist / mic | ours 2 bit/s, PESQ 4.0 | 4.3 | 4.4 | 4.5 | 4 bit/s, 4.0 | 4.3 | 6 bit/s, 4.0 | DeAR native / 4.0 / 4.3 | AudioSeal / WavMark |
  |---|---|---|---|---|---|---|---|---|---|
  | set A, laptop mic, run 1 | 0.78 | 0.43 | - | - | - | - | - | - | 0.00 / 0.00 |
  | set A, laptop mic, run 2 | 0.95 | 0.60 | - | - | - | - | - | - | 0.00 / 0.00 |
  | set A, phone mic, run 2 | 0.98 | 0.93 | - | - | - | - | - | - | 0.00 / 0.00 |
  | set B, laptop mic | 0.93 | 0.80 | 0.65 | 0.28 | 0.83 | 0.48 | 0.73 | 0.10 / 0.00 / 0.00 | - |
  | set B, phone mic | 0.98 | 0.90 | 0.88 | 0.73 | 0.95 | 0.85 | 0.93 | 0.18 / 0.00 / 0.00 | - |

  The external speakers through the laptop mic are the hardest real channel so far (alignment correlation 0.20-0.29
  against 0.37-0.41 with the built-in speakers); the repeat of set A on the same mic moved 0.78 -> 0.95 at PESQ 4.0, so
  that mic position is also sensitive to room noise (a loud event was logged during run 1). On the phone mic ours
  carries 2 bit/s at PESQ 4.4 (0.88) and 6 bit/s at PESQ 4.0 (0.93); measured information rate on that mic 11.0 bit/s
  at PESQ 4.0 and 6.6 at 4.4. DeAR reads 10-18 % of messages at its own audible strength (PESQ 2.3) and nothing at
  PESQ 4.0 or 4.3. AudioSeal and WavMark read nothing (bit error 0.39-0.51). Caveat: the CRC-only false alarm on the
  phone mic was 2-5 % (1-2 of 40 unmarked); the reported numbers hold false alarm at 1 % with a score threshold.
  Results: `results/a1_ota_watermark/1299968`, `1300639`, `1300644` (set A), `1300891`, `1300892` (set B).
- **A1 real loudspeaker / microphone, set 3: DeAR in the room** (same laptop and room, 100 host streams per arm,
  2.8 h, 700 items, new host streams; alignment correlation median 0.37). DeAR decoded with the fixed band-filtered
  decoder (seed 0); ours seed 0 for a repeat. Message success at <= 1 % false alarm, n = 100 per cell:

  | method | PESQ 4.0 | PESQ 4.3 | SNR 22 dB | SNR 26 dB |
  |---|---|---|---|---|
  | ours (CRC, list of 4) | 0.98 | 0.81 | - | - |
  | DeAR | 0.00 (bit error 0.43) | 0.00 (0.46) | 0.00 (0.30) | 0.00 (0.38) |

  Ours repeats the full-set result on new host streams (1.00 / 0.94 there; PESQ 4.3 is the weaker point, bit error
  0.09-0.12, CRC false alarm 0-1 % on 100 unmarked twins). DeAR reads nothing in the real room even when given the same
  residual energy as ours (SNR 22 / 26 dB), consistent with the simulated table above. Results:
  `results/a1_ota_watermark/1296502/real_set3.json`.
- **A1 simulated channels at matched energy** (seeds 0-2, 300 recordings each; every method scaled to the same
  host-to-watermark SNR; message success at <= 1 % false alarm, mean of 3 seeds):

  | SNR | method | clean | ota | ota_mp3 | ota_noisy |
  |---|---|---|---|---|---|
  | 22 dB | ours | 1.000 | 1.000 | 0.999 | 0.961 |
  | 22 dB | AudioSeal | 0.640 | 0.053 | 0.042 | 0.027 |
  | 22 dB | WavMark | 0.810 | 0.309 | 0.221 | 0.189 |
  | 26 dB | ours | 1.000 | 0.986 | 0.978 | 0.859 |
  | 26 dB | AudioSeal | 0.600 | 0.018 | 0.021 | 0.009 |
  | 26 dB | WavMark | 0.991 | 0.458 | 0.336 | 0.288 |

  At equal watermark energy the gap stays large. WavMark is *worse* at the louder 22 dB than at 26 dB (even on
  clean audio), so pushing it beyond its native strength does not help it; its best over-the-air number anywhere is
  0.46.
- **A1 Opus in the training channel: mixed, not adopted.** Ours retrained (3 seeds, 20k steps, same settings) with a
  straight-through Opus 8-24 kbit/s stage on half the batches, against the main-run models (seeds 0-2), simulated
  test channels, PESQ 4.0:

  | model | ota | ota_mp3 | ota_opus12 |
  |---|---|---|---|
  | main run | 0.998 | 0.993 | 0.778 +- 0.022 |
  | with Opus stage | 0.933 +- 0.059 | 0.920 | 0.603 +- 0.249 |

  Seed-dependent: seed 1 with the Opus stage reaches 1.000 on `ota` and 0.933 on `ota_opus12`, seeds 0 and 2 are
  worse than the main run on both. Training ends at bit error 0.08 against 0.03, so the harder channel probably
  needs more steps. AudioSeal and WavMark on `ota_opus12`: 0.019 / 0.003. Results: `results/a1_ota_watermark/1294596`.
- **A1 real full set, 6 dB quieter playback** (same 1000 items, laptop and room as the full set, playlist scaled by
  -6 dB; alignment correlation median 0.37). Message success at <= 1 % false alarm, n = 100:

  | method | PESQ 4.0 | PESQ 4.3 |
  |---|---|---|
  | ours (CRC, list of 4) | 0.94 (full level: 1.00) | 0.76 (0.94) |
  | AudioSeal | 0.02 | 0.00 |
  | WavMark | 0.00 | 0.00 |

  No CRC false alarm on the unmarked twins. Ours degrades gracefully with less acoustic level; the baselines were
  already at chance. Results: `results/a1_ota_watermark/1294730/real_full_m6.json`.
- **A1 training-channel ablation** (ours trained on the original simple channel only, 3 seeds, simulated test
  channels, message success at <= 1 % false alarm): `ota` 0.982 +- 0.010 / 0.853 +- 0.067 at PESQ 4.0 / 4.3 (full
  channel: 0.998 / 0.981), `ota_mp3` 0.970 / 0.817 (full: 0.993 / 0.976), held-out reverb 0.996 / 0.943. The
  realistic loudspeaker / room / microphone training channel matters mostly at the quieter strength.
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
