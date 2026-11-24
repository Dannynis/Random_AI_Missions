# A2: learned cold-start active-learning policies on frozen DINOv2 features (negative)

**2026-11-24** · code `experiments/a2_al_policy` · results `results/a2_al_policy/` · 11 GB GPUs, about 1.5 h in total

## Question
Origin: Random-Ideas PRs #16, #36. On MNIST a small MLP query policy trained with group-relative RL beat BALD and
TypiClust-style warm-up on 10/10 seeds. Does a learned policy beat the best hand-made rule on CIFAR-10/100 when every
method works on frozen self-supervised features?

## Setup
- Features: DINOv2 ViT-B/14 and ViT-S/14, frozen. Learners: linear probe and label propagation.
- Rules: Random, Entropy, Margin, Coreset (k-center), TypiClust, ProbCover (approximate: balls truncated to 100-NN), BADGE.
- Budgets 10 to 1000 labels; metric = accuracy averaged over the budget range (AUC); 10 seeds, each on its own 80 % pool.
- Policy v1: the GAL-style per-candidate MLP, group-relative RL, meta-trained on 10-way CIFAR-100 tasks (classes 0-49);
  tested on CIFAR-10, held-out CIFAR-100 classes 50-99 and full CIFAR-100.
- Policy v2: added cluster features, behaviour cloning from TypiClust, then RL fine-tuning.
- Follow-up: TypiClust then Margin / BADGE after 100 or 300 labels (TCM-style switch).

## Result
- TypiClust is the best hand-made rule in all 12 cells (2 feature sets x 2 learners x 3 task groups). ViT-B, linear
  probe: AUC 0.920 (CIFAR-10), 0.536 (CIFAR-100), 0.943 (held-out 10-way CIFAR-100).
- Policy v1 loses to TypiClust on 0/10 paired seeds in every cell (-0.045 to -0.069 AUC) and is below Random on
  CIFAR-10. Its cold start is the problem (0.39 vs 0.66 accuracy at 10 labels); from 300 labels on it matches or
  slightly exceeds every rule.
- Policy v2: behaviour cloning alone reproduces TypiClust (paired difference within +-0.0003 in all 6 cells);
  RL fine-tuning then makes it worse in all 6 cells (-0.003 to -0.016).
- Switch heuristics gain at most +0.0013 AUC. TypiClust leaves almost no headroom on these features.

## Verdict
**Fail** (pre-stated criterion: a policy trained on dataset A beats the best hand-made rule on datasets B and C).
The MNIST result does not transfer once a strong self-supervised representation is used.

## Caveats and next steps
- ProbCover is approximate and scores below its paper. CIFAR-10N (noisy labels) and ImageNet-100 were not run.
- A learned policy might still help where typicality is a poor prior (noisy labels, long-tailed pools); that would be
  a new hypothesis rather than a rescue of this one.

Figures: `results/a2_al_policy/figures/` (learning curves per feature set and learner, policy training curves, full table).
