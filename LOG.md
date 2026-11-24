# Lab log

Newest entry first.

## 2026-11-24, results so far and handoff

### Verdicts
- **A2 (learned active-learning policy): fail.** TypiClust is the best rule in all 12 cells on frozen DINOv2 features;
  the RL policy loses on 0/10 seeds (-0.045 to -0.069 AUC); imitation of TypiClust ties it and RL fine-tuning only
  hurts. Report: `reports/2026-11-24-a2-al-policy`.
- **B4 (NCA re-seed schedule): rising 2 -> 8 beats constant 2** on anytime accuracy on 10/10 seeds (+0.018); whether it
  is the timing or just the amount of fresh data is open (6/10 against constant 5). Report:
  `reports/2026-11-24-b4-nca-schedule`. Follow-up with 20 more seeds is running.

### Interim (not verdicts)
- **A1, seed 0 of 5** (300 recordings of 8 s, 16-bit messages, random start, message success at <= 1 % false alarm):

  | channel | ours, PESQ 4.0 | ours, PESQ 4.3 | AudioSeal, PESQ 4.0 | AudioSeal, PESQ 4.3 | WavMark |
  |---|---|---|---|---|---|
  | clean | 1.000 | 1.000 | 0.69 | 0.63 | 1.000 |
  | MP3 64k | 1.000 | 1.000 | 0.66 | 0.61 | 0.990 |
  | reverb (held-out rooms) | 1.000 | 1.000 | 0.03 | 0.04 | 0.500 |
  | over the air (`ota`) | 1.000 | 0.987 | 0.04 | 0.04 | 0.263 |
  | over the air + MP3 | 0.983 | 0.960 | 0.03 | 0.02 | pending |

  The WavMark column is at its native strength (PESQ 4.17); its matched-PESQ rows were still running.

  Ours uses CRC-8 with no threshold (raw false alarm 0-0.3 %). At native strength our watermark sits at PESQ 3.83
  (SI-SNR 20 dB); WavMark's native PESQ is 4.17 and AudioSeal's 4.47. AudioSeal numbers in this table come from the
  first evaluation, whose threshold folds were split by source type (realised false alarm 7-11 %, so they are if
  anything optimistic for AudioSeal); a corrected rerun of seed 0 is queued. The test channels come from the same
  simulator family as training, so the stress set of unseen channels and real recordings are needed before any claim.
- **A3, partial** (mean accuracy over 23 held-out OpenML tables and k = 1..20): TabPFN v2 0.699, untrained ridge
  scaffold 0.685, logistic regression 0.681, 0.5B generator 0.673 (3 seeds, its scaffold 0.686), 7B generator 0.665
  (seed 0, its scaffold 0.686), 7B in-context prompting 0.589. So far the generator is below its own scaffold at both
  sizes, as in the CPU-scale runs. Remaining arms are running.

### Handoff for the next session
- **Running / queued:**
  - A1 main run: seeds 0-2 (training + evaluation, about 3.5 h per seed), then seeds 3-4 in a chained job, then the
    training-channel ablation (3 seeds, ours only), all on one 80 GB GPU in sequence.
  - A1 corrected re-evaluation of seed 0, then (after seeds 3-4 exist) the stress-set evaluation of all 5 seeds,
    on 11 GB GPUs.
  - A3 remaining arms (7B seeds 1-2, 1.5B x 3, random-init control) on one 80 GB GPU.
  - B4 follow-up: seeds 10-29 of constant 2, constant 5, 2 -> 8 and the ramp end-points 2 -> 12, 1 -> 8 (two jobs).
- **Check next:**
  1. A1: use the corrected seed-0 rerun and seeds 1-4 (fixed folds; per-recording scores saved). Build the table with
     `python experiments/a1_ota_watermark/make_figures.py results/a1_ota_watermark <out>`; apply the pre-stated
     criterion on `ota` / `ota_mp3`; read the stress set and ablation; write `reports/<date>-a1-ota-watermark`.
  2. A3: when `summary.json` shows a verdict, write the report. If the job stops on its time limit, it resumes with
     the `--continue` option described in the experiment README.
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
