# Hypothesis backlog

Ranked by paper potential times how much large-scale compute changes the answer.
Origin links point at the Random-Ideas daily PRs. Status: `todo`, `running`, `done`, `dropped`.

## Tier A: paper candidates

### A1. Over-the-air audio watermarking against published neural watermarks  (`done`, pass: `reports/2026-11-24-a1-ota-watermark`; stress / real-set seeds 1-4 / DeAR still extending)
- **Origin:** Random-Ideas PRs #19, #33, #45, #57. Learned code + soft-Viterbi convolutional code + blind sync + CRC took
  simulated over-the-air message success from 0.025 to ~0.8 (16 bits in 8 s), on small models and 2-3 seeds.
- **Why scale matters:** needs a full-size encoder/decoder trained on real speech and music (LibriSpeech, FMA, MUSDB)
  through a realistic differentiable channel (measured room impulse responses, loudspeaker/microphone responses,
  noise, resampling, clock drift, MP3/Opus), and head-to-head baselines.
- **Strong test:** AudioSeal and WavMark (public checkpoints) versus our system under the same channel set, at matched
  audibility (PESQ / ViSQOL / SI-SNR), 16 and 32 bit payloads, 5 seeds. Report message success at <= 1 % false alarm,
  bit error rate, and detection AUC, per channel.
- **Pass:** ours beats both baselines on re-recorded (speaker to microphone) channels by a clear margin at equal audibility.
- **Next after the simulated run (needs the owner's hardware):** play the 300 test streams of one seed through a real
  loudspeaker and record them with a phone / laptop microphone in two rooms; rerun the same receivers on the recordings.

### A2. Learned query policies for cold-start active learning, beyond MNIST  (`done`, negative: `reports/2026-11-24-a2-al-policy`)
- **Origin:** GAL sessions and PRs #16, #36. A 15-feature MLP policy trained with group-relative RL beat BALD and
  TypiClust-style warmup on 10/10 seeds on MNIST and transferred to Fashion-MNIST.
- **Why scale matters:** CIFAR-10/100 and ImageNet-100 with a self-supervised backbone (DINOv2 features) and a
  semi-supervised learner need thousands of short training runs for policy training and evaluation.
- **Strong test:** pool-based policy versus Random, Entropy, BADGE, Coreset, TypiClust and ProbCover under the same
  semi-supervised learner; policy meta-trained on one dataset and tested on others; real noisy labels (CIFAR-10N);
  10 seeds; budgets 10 to 1000 labels.
- **Pass:** policy trained on dataset A beats the best hand-made rule on datasets B and C in label-efficiency AUC.

### A3. LLM-generated network weights with a real LLM  (`done`, fail: `reports/2026-11-24-a3-llm-weights`; 7B generator 0.668 vs its scaffold 0.687)
- **Origin:** PRs #14, #34 and the `llm_weights` project. All runs used a CPU stand-in for Qwen; a fitted ridge
  scaffold matched logistic regression and the generator hurt.
- **Why scale matters:** the actual hypothesis (an LLM conditioned on a dataset description emits useful weights)
  was never tested with an LLM. Needs Qwen2.5 / Llama-3 class models (7B and larger) with LoRA.
- **Strong test:** LoRA-tuned LLM emitting weight deltas for tabular tasks from an SCM prior, versus TabPFN,
  logistic regression, the fitted scaffold, and in-context prompting of the same LLM. Held-out real datasets (OpenML-CC18).
- **Pass:** generator beats the scaffold and in-context baseline on held-out real datasets.

## Tier B: scale checks that settle open questions

### B1. Forward-Forward: is the loss the problem, not locality?  (`running`: depth sweep done, `reports/2026-11-27-b1-ff-depth` — H1 FAIL, H2 PASS with plateau; B1b running)
- **Origin:** PR #46 (MNIST: Hinton recipe 7.26 % error, pairwise loss 1.98 %, local cross-entropy 1.49 %, backprop
  1.85 %) and PR #60 (label-free pairwise FF 2.11 %, threshold 4.03 %). Both found locality is free and the threshold
  loss is the cost, but on MNIST **depth adds nothing** for a local learner.
- **Strong test:** CIFAR-10/100 with conv blocks; bp vs greedy local-CE vs goodness/pairwise-FF at equal arch and
  epochs, depth sweep {4,8,12}, 3 seeds. Two pre-stated questions: H1 parity (best local within 3 pp of bp on
  CIFAR-10), H2 composition (deepest-block probe beats first by ≥5 pp — does locality compose with depth on a hard
  task, unlike MNIST?). Prior work: Belilovsky et al. greedy layerwise (ICML 2019), Nøkland & Eidnes local errors.
- **Result (2026-11-27, 54 cells):** H1 FAIL — greedy local CE trails backprop by 12.5/16.4/16.7 pp on CIFAR-10 at
  depth 4/8/12 (CIFAR-100: 14.5-23.3 pp). H2 PASS as stated (+33 pp block 1 → best) but local CE **plateaus at ~block
  7** while bp keeps composing to block 10-12. FF goodness (pairwise) is far worse at conv depth (0.43-0.46 CIFAR-10,
  peaks at block 6 then degrades). The MNIST finding "locality is free" does not carry to CIFAR.
- **B1b (running):** does a stronger local objective (Belilovsky aux head `local_aux`, InfoPro-style reconstruction
  `local_rec`) close the gap and lift the plateau? Pre-stated: within 3 pp of bp at d8 CIFAR-10; block 12 beats block 6
  by ≥2 pp.
- **Possible LLM-scale sequel:** greedy local learning for language-model pretraining (arXiv 2610.04867 studies the
  gap and objective design) — check that paper before building.

### B2. Differentiable logic-gate networks at depth  (`todo`)
- **Origin:** PR #56. Straight-through forward took depth-12 training from 10 % to 86.7 %.
- **Strong test:** CIFAR-10 at the published network sizes, depth 6 to 48, versus the published residual-init
  and Gumbel variants. Check prior work first (a 2025 paper may cover this).

### B3. Mixture-of-experts routing versus dense at real scale  (`todo`)
- **Origin:** PRs #3, #20, #40, #52. Learned routers recover 24-82 % of oracle gain; self-supervised features in a
  dense model beat every router.
- **Strong test:** small transformer language model (GPT-2 small scale) with MoE layers on a multi-domain corpus,
  dense at equal compute, oracle domain routing as the ceiling.

### B4. Neural cellular automata: persistence schedule  (`done`: `reports/2026-11-24-b4-nca-schedule`)
- **Origin:** PRs #28, #44, #58. Reversed re-seed schedule (2 to 8) was best on 1 seed.
- **Strong test:** 5 seeds, plus mid-switch schedule; CIFAR-10 / larger canvases. Cheap.
- **Result (30 seeds):** H1 confirmed — rising 2 -> 8 beats constant 2 on anytime accuracy 30/30 seeds (+0.019,
  CI [+0.014, +0.025]). H2 (timing not amount) not confirmed by the pre-stated 80%-of-seeds bar: 2 -> 8 beats
  equal-data const5 on only 21/30 (70%), though the mean gain is real (+0.017, CI excludes 0); const5 ties const2.
  A lower-starting wider ramp 1 -> 8 is the best arm (ties 2 -> 8, beats const2 18/20); 2 -> 12 overshoots and hurts
  persistence. The rising *shape* from a low start is what helps.

### B5. Network growth controller versus size sweep at CIFAR scale  (`todo`)
- **Origin:** nn_builder sessions, PRs #15, #32, #54. Ties a 5-size sweep at one third of the compute on toy tasks.
- **Strong test:** ResNet / small ViT width-depth growth on CIFAR-10/100, versus ASHA at equal total compute,
  3-5 controller training seeds.

### B6. Adversarial audio evaluation: simulated versus real channel transfer  (`todo`, manual)
- **Origin:** PR #13. Not to be run automatically; listed for the owner to drive.

### B7. Learned query policies where typicality is a poor prior  (`running`: step 1 PASS, `reports/2026-11-26-b7-hard-al`; policy build justified)
- **Origin:** A2 (2026-11-24): on clean CIFAR with DINOv2 features TypiClust leaves about 0.001 AUC of headroom and no
  learned policy beats it. Typicality should fail on noisy labels (CIFAR-10N) and long-tailed pools.
- **Strong test:** the A2 pipeline on CIFAR-10N (worst label set) and a long-tailed CIFAR-100 pool, 10 seeds; policy
  meta-trained on synthetic label noise only.
- **Step 1 result (2026-11-26, precondition test):** PASS. On long-tailed CIFAR-100, Coreset beats TypiClust by +0.063
  AUC (lp, 10/10 seeds) / +0.028 (linear); on 40% symmetric noise, Margin/BADGE beat it by +0.045/+0.034 (linear,
  8/10). Under 40% *asymmetric* noise no rule clears the 0.01 bar — TypiClust holds there. So typicality is a poor
  prior exactly where coverage (long-tail) or uncertainty (symmetric noise) is the right geometry. The policy build is
  justified but its real bar is DIRECT (arXiv 2312.09196) and the ICML-2026 foundation-prior AL method, not just the
  generic rules.

### B8. Exemplar selection for a memory (nearest-neighbour) classifier at scale  (`todo`, low GPU need)
- **Origin:** Random-Ideas PR #62 (`2026-11-27-memory-exemplar-select`). On 47-way class-incremental EMNIST with a
  frozen 93k-param encoder, k-center (k-means medoid) writes beat random by +0.062 at 20/class and +0.127 at 5/class
  (5 seeds); a readout-optimised swap selector overfits and doubles forgetting.
- **Strong test:** same protocol on class-incremental ImageNet-100 / CIFAR-100 with a frozen foundation encoder
  (DINOv2 / CLIP), budgets 1-40 per class, versus random, herding, and the pretrained-feature CIL baselines
  (prototype / RanPAC-style). Mostly feature extraction plus kNN — small tier. Check prior work on coreset selection
  for prototype/kNN CIL first.

## Tier C: cheap sweeps
- Seeds and ablations for any Tier A/B experiment.
- B4 in full.
- Dataset-distillation label ablations (PRs #38, #50) on CIFAR-10.
