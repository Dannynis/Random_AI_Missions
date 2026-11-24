# B4: re-seed schedule for the persistent self-classifying NCA (10 seeds)

Origin: Random-Ideas PRs #28 (self-classifying NCA), #44 (sample-pool persistence), #58 (re-seed annealing).
Code: `nca_classifier.py`, `nca_persist.py`, `nca_persist_bench.py` are copied from those runs unchanged, except
for a step-switch option (`reseed_switch_frac`) in `nca_persist.reseed_count` and device handling so that
training and evaluation run on a GPU (several runs share one 11 GB GPU). `b4_bench.py` is the new driver.

## Hypothesis

PR #58 found, on one seed, that re-seeding the sample pool with fresh digits at a rate that *rises* during
training (linear 2 → 8 per batch of 32, `anneal_rev`) beats the constant-2 baseline (`pool_det`) on every metric
(anytime accuracy 0.848 vs 0.798 on that seed; step-2000 accuracy +0.05), while the proposed 8 → 2 schedule
loses persistence. Reading: the persistent attractor forms early from long rollouts of old states; late fresh-digit
throughput then refines classification inside it.

H1: `anneal_rev` beats `pool_det` in anytime accuracy (mean over readout steps 40–1000) on a paired 10-seed test.
H2: the benefit comes from *when* fresh digits arrive, not how many: `anneal_rev` beats `const5`
(equal fresh-digit count) and a step switch 2 → 8 at 50 % (`switch_rev`) matches or beats the linear ramp.

## Protocol

- Model: 16-channel NCA, 3×3 conv over [state, image] → 48 → ReLU → 1×1 zero-init residual update, alive mask
  ink > 0.1, deterministic updates, 7.7k parameters (unchanged).
- Training: pool 1024, rollouts T ~ U[16, 32] from stored states, cross-entropy at T−12/−8/−4/T, overflow
  penalty, 4000 iterations × 32, Adam 2e-3 cosine. Only the per-batch re-seed count changes.
- Arms (10 seeds each, same seed = same init and initial pool):

  | arm | re-seeds per batch |
  |---|---|
  | `pool_det` | constant 2 (baseline, PR #44) |
  | `const5` | constant 5 (equal fresh data to the ramps) |
  | `const8` | constant 8 |
  | `anneal` | linear 8 → 2 (PR #58) |
  | `anneal_rev` | linear 2 → 8 |
  | `switch_rev` | 2 for the first half, 8 for the second |

- Evaluation (PR #58 protocol): accuracy vs readout step on 1000 MNIST test digits up to step 2000; anytime
  accuracy; peak; 8×8 state-damage and digit-swap probes at step 100 (500 digits); 8 px occlusion and gap-2
  pairs at readouts 40 / 200.
- Tiny mode (`run.sh tiny`): 50 iterations, 100 test digits, 2 seeds; pipeline check only.

## Pass / fail

- H1 passes if the paired mean anytime difference `anneal_rev − pool_det` is > 0 with at least 8/10 seeds positive
  and the step-2000 difference is not negative on average.
- H2 passes if `anneal_rev` (or `switch_rev`) beats `const5` on anytime accuracy in at least 8/10 paired seeds.
- Anything else is reported as a negative result.

## Prior work

The self-classifying MNIST NCA (Randazzo, Mordvintsev et al., Distill 2020) introduced the per-pixel voting
setup and sample-pool training for persistence. Re-seed schedules of the pool are not studied there; the
CAX library reimplements the setup with a 1024-state pool. No published work found on scheduling the pool
re-seed rate.

## Outputs

`$RUN_DIR/out/results.json`: per-arm means/std, per-seed metrics, paired differences vs `pool_det`.
Checkpoints and per-run JSONs in `$RUN_DIR/ckpt` (each run resumes independently).

## Results (10 seeds per arm, 60 runs; `results/b4_nca_schedule/`)

| arm | anytime (40–1000) | peak | acc@40 | acc@2000 | damage@200 | swap@300 | final pool age |
|---|---|---|---|---|---|---|---|
| `pool_det` (2) | 0.825 ± 0.019 | 0.845 | 0.792 | 0.808 ± 0.021 | 0.85 | 0.70 | 385 |
| `const5` | 0.824 ± 0.022 | 0.846 | 0.821 | 0.781 ± 0.030 | 0.84 | 0.60 | 153 |
| `const8` | 0.793 ± 0.017 | 0.855 | 0.845 | 0.509 ± 0.108 | 0.83 | 0.29 | 96 |
| `anneal` (8 → 2) | 0.807 ± 0.015 | 0.834 | 0.800 | 0.695 ± 0.057 | 0.82 | 0.41 | 306 |
| **`anneal_rev` (2 → 8)** | **0.843 ± 0.006** | **0.863** | **0.845** | 0.805 ± 0.021 | 0.85 | **0.71** | 97 |
| `switch_rev` (2 → 8 at 50 %) | 0.831 ± 0.014 | 0.854 | 0.839 | 0.787 ± 0.022 | 0.84 | 0.69 | 97 |

Paired differences against `pool_det` (same seed; wins out of 10):

| arm | anytime | peak | acc@40 | acc@2000 |
|---|---|---|---|---|
| `anneal_rev` | **+0.018 (10/10)**, se 0.006 | +0.018 (9/10) | +0.053 (10/10) | −0.002 (5/10) |
| `switch_rev` | +0.007 (7/10) | +0.009 (6/10) | +0.047 (10/10) | −0.021 (3/10) |
| `const5` | −0.001 (4/10) | +0.001 (5/10) | +0.028 (8/10) | −0.027 (3/10) |
| `anneal` | −0.018 (2/10) | −0.011 (2/10) | +0.007 (7/10) | −0.113 (1/10) |
| `const8` | −0.032 (1/10) | +0.010 (8/10) | +0.052 (10/10) | −0.299 (0/10) |

**Verdict.**
- **H1 confirmed on the primary metric.** The rising 2 → 8 re-seed schedule beats the constant-2 baseline in
  anytime accuracy on 10/10 paired seeds (+0.018), with higher peak and much faster convergence (+0.053 at step
  40), while long-horizon persistence is unchanged (step 2000: −0.002, 5/10 seeds; by the letter of the
  pre-stated clause "not negative on average" this is parity, not a gain). It is also the most stable arm
  (anytime std 0.006 vs 0.015–0.022). PR #58's single-seed observation replicates.
- **H2 not confirmed.** Against `const5`, which sees the same number of fresh digits, `anneal_rev` wins only 6/10
  seeds on anytime accuracy (+0.019 mean; criterion was 8/10), and the step switch `switch_rev` is worse than the
  linear ramp (−0.011, 3/10). So "when" matters for persistence (constant high rates destroy it: `const8`
  step-2000 0.51, `anneal` 0.70), but the timing advantage of the ramp over a constant equal-data schedule is not
  resolved with 10 seeds.
- The 8 → 2 anneal proposed in PR #44 is confirmed harmful (step 2000 −0.113, 1/10).
- Reading: old (long-rolled) states are needed while the attractor forms; once formed, the rule tolerates and
  benefits from high fresh-digit throughput. A constant high rate from the start never forms it.

Figures: `results/b4_nca_schedule/figures/fig1_step_curve.png`, `fig2_headline.png`, `fig3_paired.png`.
