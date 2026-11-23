# Lab log

Newest entry first.

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
