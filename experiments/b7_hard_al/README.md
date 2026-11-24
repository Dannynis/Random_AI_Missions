# B7 (step 1): does typicality lose its edge when it is a poor prior?

Origin: A2 (`reports/2026-11-24-a2-al-policy`). On clean CIFAR with frozen DINOv2 features, TypiClust is the best
cold-start active-learning rule in all 12 cells and no learned policy beats it — it leaves ~0.001 AUC of headroom.
A2's report proposed B7: typicality should be a *poor* prior where "typical" points are the wrong thing to label —
long-tailed pools (typical points over-represent head classes) and heavy label noise (typical points can be
confidently mislabeled). This experiment is the precondition test: **before** spending strong-GPU time meta-training a
noise/imbalance-aware policy, check whether any existing rule already beats TypiClust on these hard pools. If none does,
B7 inherits A2's negative and the policy build is not pursued.

## Hypothesis
On at least one hard regime, a label-/geometry-aware existing rule beats TypiClust on test-accuracy label-efficiency
AUC, opening headroom that a learned policy could exploit.

## Setup
Reuses the A2 pipeline unchanged (`experiments/a2_al_policy/al_lib.py`, same frozen DINOv2 ViT-B/14 features from the
shared cache, same learners and budget schedule, same paired-seed protocol). Only the pool construction changes.
This driver (`b7_bench.py`) trains **no policy**; it runs the seven A2 baseline rules (Random, Entropy, Margin,
Coreset, TypiClust, ProbCover, BADGE) on three hard eval tasks.

- **`c100_longtail`** — CIFAR-100 pool subsampled to exponential class imbalance (largest/smallest = 100; the head/tail
  class order is reshuffled per seed). Test set left balanced, so overall test accuracy is a tail-sensitive metric.
- **`c10_sym40`** — CIFAR-10 pool with 40 % symmetric label noise (flip to a uniformly random different class). Test
  labels are clean. The learner is trained on the noisy queried labels.
- **`c10_asym40`** — CIFAR-10 pool with 40 % asymmetric noise (c -> (c+1) mod 10, a fixed confusable mapping).

Budgets 10..1000 labels; two learners (linear probe `linear`, label propagation `lp`); 10 seeds per (task, rule),
paired (same seed = same pool subsample and draws for every rule).

## Pass / fail (stated before the run)
**Headroom (pass):** some existing rule beats TypiClust by >= 0.01 absolute AUC on >= 8/10 paired seeds, on at least one
(task, readout). Then B7's policy build is justified.
**No headroom (fail):** TypiClust is within 0.01 of the best rule on every hard task and readout. Then B7 inherits A2's
negative result and the learned-policy build is dropped. Either outcome is reported.

## Prior work
TypiClust (Hacohen et al., ICML 2022) and ProbCover (Yehuda et al., 2022) are the low-budget density/coverage rules.
Active learning under class imbalance and under label noise are each studied (e.g. class-balanced and
robust-acquisition variants), but the specific question here — whether the *clean-data-best* density rule keeps its
edge on DINOv2 features under long tails and heavy noise, as the bar a learned policy must clear — follows directly
from A2 and is cheap to settle first. A quick literature check is logged with the verdict.

## Outputs
`$RUN_DIR/out/summary.json`: per-(task, rule, readout) mean/std AUC over seeds, each rule's paired contrast vs
TypiClust (mean, win fraction), and the list of headroom cells. `run.sh tiny` is a 2-seed pipeline check on
small features.
