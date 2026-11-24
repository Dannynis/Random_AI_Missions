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
- **H2 (timing beats the same amount of data spent at a constant rate): settled at 30 seeds, does not meet the
  pre-stated bar, but the mean effect is real.** Pooling seeds 0-29 (30 paired seeds), 2 -> 8 beats constant 5 on
  anytime accuracy by +0.0173 (95 % bootstrap CI [+0.0083, +0.0267], excludes 0) but on only 21/30 seeds (70 %),
  below the pre-registered 80 %-of-seeds criterion. So the ramp carries a small but statistically distinguishable
  timing advantage over spending the same fresh data at a constant rate, yet it is too inconsistent per-seed to call
  H2 confirmed as written. Constant 5 itself is at parity with constant 2 (+0.002, 15/30), so a constant rate buys
  nothing; the gain is specific to the rising shape. The step switch at 50 % was worse than the linear ramp (10-seed
  only: -0.011, 3/10).
- **Ramp shape (seeds 10-29, 20 seeds each).** A *wider, lower-starting* ramp is as good or better: 1 -> 8 matches
  2 -> 8 (anytime 0.844 vs 0.844) and beats const5 on 15/20 (CI excludes 0) and const2 on 18/20; 2 -> 12 is worse
  (0.836, 12/20 vs const5, CI includes 0) and loses persistence (acc@2000 0.778). Reaching a high rate *gradually
  from a low start* is what helps; overshooting the end rate hurts.
- The original 8 -> 2 anneal is harmful (step 2000 -0.113, 1/10); constant 8 destroys persistence (0.51 at step 2000).

## Verdict
Rising schedule: **pass** against the origin's baseline (anytime +0.019, 30/30 seeds; CI [+0.014, +0.025]).
Timing-versus-amount (H2): **not confirmed** by the pre-stated 80 %-of-seeds bar (70 % at 30 seeds), though the mean
timing gain over equal-data const5 is real (CI excludes 0). A lower-starting wider ramp (1 -> 8) is the best arm.

## Next
H2 is now settled; no more seeds needed for this contrast. Open follow-ups if revisited: larger canvases / CIFAR
(does the timing effect grow when the attractor is harder to form?) and 1 -> 8 vs 1 -> 6 to find where the wider ramp
peaks. Cheap; low priority given the modest effect size.

Figures: `results/b4_nca_schedule/figures/` (step curves, headline bars, paired differences;
`fig4_h2_30seeds.png` is the 30-seed H2 test).
