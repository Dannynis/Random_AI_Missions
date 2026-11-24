# Hypothesis backlog

Ranked by paper potential times how much large-scale compute changes the answer.
Origin links point at the Random-Ideas daily PRs. Status: `todo`, `running`, `done`, `dropped`.

## Tier A: paper candidates

### A1. Over-the-air audio watermarking against published neural watermarks  (`running`: `experiments/a1_ota_watermark`)
- **Origin:** Random-Ideas PRs #19, #33, #45, #57. Learned code + soft-Viterbi convolutional code + blind sync + CRC took
  simulated over-the-air message success from 0.025 to ~0.8 (16 bits in 8 s), on small models and 2-3 seeds.
- **Why scale matters:** needs a full-size encoder/decoder trained on real speech and music (LibriSpeech, FMA, MUSDB)
  through a realistic differentiable channel (measured room impulse responses, loudspeaker/microphone responses,
  noise, resampling, clock drift, MP3/Opus), and head-to-head baselines.
- **Strong test:** AudioSeal and WavMark (public checkpoints) versus our system under the same channel set, at matched
  audibility (PESQ / ViSQOL / SI-SNR), 16 and 32 bit payloads, 5 seeds. Report message success at <= 1 % false alarm,
  bit error rate, and detection AUC, per channel.
- **Pass:** ours beats both baselines on re-recorded (speaker to microphone) channels by a clear margin at equal audibility.

### A2. Learned query policies for cold-start active learning, beyond MNIST  (`done`, negative: `reports/2026-11-24-a2-al-policy`)
- **Origin:** GAL sessions and PRs #16, #36. A 15-feature MLP policy trained with group-relative RL beat BALD and
  TypiClust-style warmup on 10/10 seeds on MNIST and transferred to Fashion-MNIST.
- **Why scale matters:** CIFAR-10/100 and ImageNet-100 with a self-supervised backbone (DINOv2 features) and a
  semi-supervised learner need thousands of short training runs for policy training and evaluation.
- **Strong test:** pool-based policy versus Random, Entropy, BADGE, Coreset, TypiClust and ProbCover under the same
  semi-supervised learner; policy meta-trained on one dataset and tested on others; real noisy labels (CIFAR-10N);
  10 seeds; budgets 10 to 1000 labels.
- **Pass:** policy trained on dataset A beats the best hand-made rule on datasets B and C in label-efficiency AUC.

### A3. LLM-generated network weights with a real LLM  (`running`: `experiments/a3_llm_weights`)
- **Origin:** PRs #14, #34 and the `llm_weights` project. All runs used a CPU stand-in for Qwen; a fitted ridge
  scaffold matched logistic regression and the generator hurt.
- **Why scale matters:** the actual hypothesis (an LLM conditioned on a dataset description emits useful weights)
  was never tested with an LLM. Needs Qwen2.5 / Llama-3 class models (7B and larger) with LoRA.
- **Strong test:** LoRA-tuned LLM emitting weight deltas for tabular tasks from an SCM prior, versus TabPFN,
  logistic regression, the fitted scaffold, and in-context prompting of the same LLM. Held-out real datasets (OpenML-CC18).
- **Pass:** generator beats the scaffold and in-context baseline on held-out real datasets.

## Tier B: scale checks that settle open questions

### B1. Forward-Forward: is the loss the problem, not locality?  (`todo`)
- **Origin:** PR #46 (MNIST: Hinton recipe 7.26 % error, pairwise loss 1.98 %, local cross-entropy 1.49 %, backprop 1.85 %).
- **Strong test:** CIFAR-10/100 and ImageNet-100 with conv and ViT blocks; local losses versus backprop at equal
  epochs, 3 seeds; depth sweep.

### B2. Differentiable logic-gate networks at depth  (`todo`)
- **Origin:** PR #56. Straight-through forward took depth-12 training from 10 % to 86.7 %.
- **Strong test:** CIFAR-10 at the published network sizes, depth 6 to 48, versus the published residual-init
  and Gumbel variants. Check prior work first (a 2025 paper may cover this).

### B3. Mixture-of-experts routing versus dense at real scale  (`todo`)
- **Origin:** PRs #3, #20, #40, #52. Learned routers recover 24-82 % of oracle gain; self-supervised features in a
  dense model beat every router.
- **Strong test:** small transformer language model (GPT-2 small scale) with MoE layers on a multi-domain corpus,
  dense at equal compute, oracle domain routing as the ceiling.

### B4. Neural cellular automata: persistence schedule  (`done` for H1, H2 follow-up `running`: `reports/2026-11-24-b4-nca-schedule`)
- **Origin:** PRs #28, #44, #58. Reversed re-seed schedule (2 to 8) was best on 1 seed.
- **Strong test:** 5 seeds, plus mid-switch schedule; CIFAR-10 / larger canvases. Cheap.

### B5. Network growth controller versus size sweep at CIFAR scale  (`todo`)
- **Origin:** nn_builder sessions, PRs #15, #32, #54. Ties a 5-size sweep at one third of the compute on toy tasks.
- **Strong test:** ResNet / small ViT width-depth growth on CIFAR-10/100, versus ASHA at equal total compute,
  3-5 controller training seeds.

### B6. Adversarial audio evaluation: simulated versus real channel transfer  (`todo`, manual)
- **Origin:** PR #13. Not to be run automatically; listed for the owner to drive.

## Tier C: cheap sweeps
- Seeds and ablations for any Tier A/B experiment.
- B4 in full.
- Dataset-distillation label ablations (PRs #38, #50) on CIFAR-10.
