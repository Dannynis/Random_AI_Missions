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
- 10 seeds per (task, method). Each seed uses a different random 80 % of the training split as the pool (so
  deterministic rules such as ProbCover also have seed variance); same seed = same pool and random draws for
  every method, which makes the comparisons paired.
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

- Learned policy v2 (`run.sh policy2`), added after v1's first evaluation seeds came back 0.05 AUC below
  TypiClust: the same MLP gets three extra per-round cluster features (k-means with |L|+B clusters: cluster still
  unlabeled, cluster size, point is the densest unlabeled point of its cluster), so TypiClust is expressible
  exactly. It is first behaviour-cloned from TypiClust's ordered picks (100 episodes, Plackett-Luce likelihood),
  then fine-tuned with the same GRPO objective (200 iterations, lr 1e-3). The BC-only policy (`policy2_bc`) is
  evaluated as well, so any gain can be attributed to the RL stage. A quick sanity check on two held-out
  10-way tasks showed the BC policy matching TypiClust (AUC 0.909 vs 0.908 and 0.902 vs 0.902, 1 seed).

## Pass / fail

The policy (v1 or v2; v2 must also beat its own BC initialisation) passes if, on CIFAR-10 *and* on the held-out CIFAR-100 10-way tasks, its AUC beats the best baseline
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

## Results, stage 1 (10 seeds, budgets 10–1000; full table: `results/a2_al_policy/figures/table.md`)

AUC over the log-budget schedule, mean ± std over 10 seeds (per-seed 80 % pools). Held-out = mean of the five
disjoint 10-way CIFAR-100 tasks from classes 50–99.

| DINOv2 ViT-B/14, linear probe | CIFAR-10 | CIFAR-100 | held-out 10-way |
|---|---|---|---|
| TypiClust | **0.920 ± 0.008** | **0.536 ± 0.005** | **0.943 ± 0.003** |
| best other baseline | 0.889 (Margin) | 0.475 (ProbCover) | 0.926 (ProbCover) |
| Random | 0.879 ± 0.014 | 0.421 ± 0.010 | 0.883 ± 0.007 |
| learned policy v1 | 0.858 ± 0.016 | 0.468 ± 0.008 | 0.897 ± 0.004 |
| v1 − TypiClust (paired) | −0.062, 0/10 wins | −0.069, 0/10 | −0.045, 0/10 |

The same ordering holds with label propagation and with ViT-S/14 features: TypiClust is the best rule in every
one of the 12 (feature × learner × task) cells, by 0.015–0.07 AUC.

- **Verdict for policy v1: fails** the pre-stated criterion (0/10 wins in every cell). It is below Random on
  CIFAR-10. Its curve shows why: at 10 labels it reaches 0.39 on CIFAR-10 (TypiClust 0.66, Random 0.51), but from
  300 labels on it matches or slightly exceeds every rule (0.983 vs 0.981 for TypiClust at 1000). The 11 v1
  features cannot express "most typical point of an unlabeled cluster at the |L|+B scale", which is what wins the
  cold start. GRPO training reward rose from 0.845 to 0.895 (first vs last 20 iterations), so it learned
  something, but not the cold-start rule.
- Caveat: our ProbCover is approximate (δ-balls truncated to the 100 nearest neighbours, δ by a 95 % purity
  rule) and is weaker here than in its paper; TypiClust follows the paper's recipe more closely.

## Results, policy v2 (10 seeds, DINOv2 ViT-B/14)

Paired AUC differences against TypiClust (wins out of 10 seeds):

| | CIFAR-10 linear | CIFAR-100 linear | held-out linear | CIFAR-10 LP | CIFAR-100 LP | held-out LP |
|---|---|---|---|---|---|---|
| v2 after behaviour cloning only | +0.0000 (5) | −0.0003 (2) | +0.0002 (7) | +0.0000 (3) | −0.0001 (4) | +0.0002 (8) |
| v2 after GRPO fine-tuning | −0.0069 (2) | −0.0071 (0) | −0.0032 (2) | −0.0161 (0) | −0.0033 (0) | −0.0055 (0) |

- **Verdict for policy v2: fails.** Behaviour cloning reproduces TypiClust to within ±0.0003 AUC, so the
  architecture can express the best rule. GRPO fine-tuning (200 iterations, reward = linear-probe AUC over
  10–160 labels on 10-way tasks from CIFAR-100 classes 0–49) then moves it *below* TypiClust in all 6 cells and
  below its own initialisation (training reward 0.911 → 0.916, first vs last 20 iterations: essentially flat).
- What the RL stage changes: on CIFAR-10 (linear) it loses 0.5–3.3 points at 20–150 labels (0.771 vs 0.804 at 20) and
  gains slightly at 300–1000 (0.984 vs 0.981 at 1000). Policy v1 showed the same high-budget edge. This points
  at a simpler hypothesis for a follow-up: TypiClust for the cold start, then switch to an uncertainty or learned
  rule after a few hundred labels (cf. the TCM heuristic), tested with a reward that weights high budgets.
- Overall A2 status: on frozen DINOv2 features, no learned policy beats TypiClust; the MNIST result from the
  daily runs does not transfer. Negative result, 10 seeds, pre-stated criterion.

## Follow-up: TypiClust → uncertainty switch (TCM-style), 10 seeds

Paired AUC difference vs TypiClust (wins / 10), DINOv2 ViT-B/14:

| | CIFAR-10 lin | CIFAR-100 lin | held-out lin | CIFAR-10 LP | CIFAR-100 LP | held-out LP |
|---|---|---|---|---|---|---|
| TypiClust → Margin at 100 | +0.0000 (7) | −0.0133 (0) | +0.0009 (9) | −0.0056 (0) | −0.0147 (0) | −0.0020 (0) |
| TypiClust → Margin at 300 | −0.0003 (0) | −0.0022 (0) | +0.0006 (10) | −0.0013 (0) | −0.0024 (0) | +0.0002 (7) |
| TypiClust → BADGE at 100 | +0.0005 (8) | −0.0125 (0) | +0.0013 (10) | −0.0036 (0) | −0.0135 (0) | −0.0012 (0) |

Switching late gives the same small high-budget gain the learned policies showed (CIFAR-10 linear at 1000
labels: 0.983 vs 0.981) and is consistently positive on the 10-way tasks with the linear probe, but the AUC gain is
≤ 0.0013, and it hurts on 100-class CIFAR-100 and with label propagation. Over this budget range, TypiClust on
DINOv2 features leaves almost no room for a better query rule: the best achievable improvement found by any
method here is about +0.001 AUC.
