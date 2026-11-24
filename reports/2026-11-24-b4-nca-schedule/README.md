# B4: re-seed schedule for the persistent self-classifying NCA (mostly positive)

**2026-11-24** · code `experiments/b4_nca_schedule` · results `results/b4_nca_schedule/` · 11 GB GPU, about 2 h

## Question
Origin: Random-Ideas PRs #28, #44, #58. A persistent-pool NCA that classifies MNIST digits re-seeds a few pool
states with fresh digits every step. On one seed, a rising re-seed count (2 to 8) was best. Does it hold over 10 seeds,
and is it the timing or just the amount of fresh data?

## Setup
6 schedules x 10 seeds (60 runs), the origin's training and evaluation protocol: constant 2 (the origin's baseline),
constant 5 (same amount of fresh data as the ramps), constant 8, linear 8 -> 2, linear 2 -> 8, step 2 -> 8 at 50 %.
Metrics: anytime accuracy (mean over steps 40-1000), peak, accuracy at step 40 and step 2000 (persistence).

## Result
- **H1 (2 -> 8 beats constant 2): confirmed on anytime accuracy**, 10/10 paired seeds, 0.843 vs 0.825
  (+0.018, se 0.006); peak 0.863 vs 0.845; step 40 +0.053; lowest seed variance of all arms. Persistence at step
  2000 is at parity (-0.002, 5/10), which misses the "no worse and better" wording of the pre-stated persistence clause,
  so it is reported as no change.
- **H2 (timing beats the same amount of data spent at a constant rate): not confirmed.** 2 -> 8 beats constant 5 on
  6/10 seeds (+0.019 mean, 8/10 required). The step switch at 50 % is worse than the linear ramp (-0.011, 3/10).
- The original 8 -> 2 anneal is harmful (step 2000 -0.113, 1/10); constant 8 destroys persistence (0.51 at step 2000).

## Verdict
Rising schedule: **pass** against the origin's baseline. The timing-versus-amount question is **open**.

## Next
Running now: 20 more seeds (seeds 10-29) of constant 2, constant 5 and 2 -> 8, plus the ramp end-points 2 -> 12 and
1 -> 8, to settle H2 with 30 paired seeds.

Figures: `results/b4_nca_schedule/figures/` (step curves, headline bars, paired differences).
