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
