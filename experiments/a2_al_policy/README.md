# A2: a learned query policy for cold-start active learning on CIFAR-10/100

Origin: Random-Ideas GAL sessions (PRs #16, #36 and the GAL lab). On MNIST a small MLP policy over ~15
per-candidate features, trained with step-level / group-relative RL, beat BALD and a TypiClust-style warm-up on
10/10 seeds and transferred to Fashion-MNIST. This experiment asks whether that survives a real benchmark.

## Hypothesis

A query policy meta-trained on one family of tasks (random 10-way tasks from CIFAR-100 classes 0–49) beats the
best hand-made rule on *other* datasets (CIFAR-10, disjoint CIFAR-100 classes 50–99, full CIFAR-100) in
label-efficiency AUC, with the same frozen self-supervised features and the same learner.

## Protocol

- Features: frozen DINOv2 (ViT-B/14 main, ViT-S/14 ablation), CLS token concatenated with the mean patch token,
  images upsampled 32 → 224, L2-normalised. `extract.py` caches them once.
- Pool = training split (labels hidden until queried), test = test split.
- Learners, both trained after every round on the labeled set only:
  - linear probe: multinomial logistic regression (L-BFGS, weight decay 1e-3);
  - label propagation over the 20-NN cosine graph of the pool (semi-supervised), read out on the test set by
    kNN over the propagated pool labels.
- Budget schedule (cumulative labels): 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000.
  AUC = mean accuracy over the schedule, trapezoid in log(budget) (each doubling weighs equally).
- 10 seeds per (task, method). Same seed = same random draws for every method.
- Baselines (all in `al_lib.py`): Random; Entropy; Margin; Coreset (greedy k-center); TypiClust (|L|+B
  k-means clusters, most typical point of the largest least-labeled cluster); ProbCover (δ-ball max coverage,
  δ chosen by 95 % ball purity); BADGE (k-means++ on gradient embeddings, computed in factored form).
  Model-based rules (Entropy, Margin, BADGE) take a random first batch.
- Learned policy: MLP (11 features → 64 → 64 → 1) scores every candidate; features are entropy, margin, max
  probability, distance to the labeled set, kNN density, whether its k-means region is unlabeled, region size,
  labeled share of its predicted class, budget progress, ProbCover coverage gain and gradient-embedding norm.
  A batch is built in 4 sub-steps (features recomputed between them). Training: GRPO, groups of 8 sampled
  episodes (Gumbel-top-k = Plackett-Luce) on the same task and seed, reward = linear-probe AUC over
  10–160 labels, z-scored within the group; 150 iterations. Evaluation is greedy.

## Pass / fail

The policy passes if, on CIFAR-10 *and* on the held-out CIFAR-100 10-way tasks, its AUC beats the best baseline
for that task (chosen after the fact, which favours the baselines) by a paired mean > 0 with at least 8/10 seed
wins, for at least one of the two learners. Full CIFAR-100 is reported as a harder transfer test.
Otherwise the result is negative and is reported as such.

## Prior work (quick search)

- Learned / meta-learned AL policies exist (Konyushkova et al. 2017 "Learning Active Learning from Data";
  Fang et al. 2017; Ravi & Larochelle 2018 meta-learned batch AL), mostly from random-init models and without
  self-supervised features; transfer across datasets was weak.
- Low-budget rules on self-supervised features: TypiClust (Hacohen et al. 2022) and ProbCover (Yehuda et al.
  2022) beat uncertainty sampling at small budgets; a 2024 workshop paper (TCM) switches from TypiClust to
  margin sampling. These are the baselines a learned policy has to beat; no published learned policy was
  found that is compared against them on frozen DINOv2 features.

## Status / scope of this first version

Stage 1 (this directory): feature extraction, all baselines, first policy version. CIFAR-10N noisy labels,
ImageNet-100 and a fine-tuned semi-supervised learner (e.g. FlexMatch) are follow-ups.

## Running

`run.sh tiny` (512-image subsets, 2 seeds), `run.sh baselines dinov2-base`, `run.sh policy dinov2-base`.
Results: `$RUN_DIR/out/summary.json` (per task × method mean/std AUC, curves, per-seed AUC, policy training log).
