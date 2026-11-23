# A3: LLM-generated network weights with a real open LLM

Backlog item A3 (`HYPOTHESES.md`). Origin: Random-Ideas PRs #14 and #34 and the owner's `llm_weights` project
(Part V `TabHyper`). Every earlier daily run used a small from-scratch transformer as a CPU stand-in for the LLM.
PR #34 found that a closed-form ridge read-out ("fitted scaffold") reaches logistic-regression accuracy
on 18 held-out OpenML tables (0.711 vs 0.707), and that the stand-in generator's deltas on top of it *hurt*
(-0.012 to -0.021), because what it learned on the synthetic prior did not transfer.

## Hypothesis

A pretrained LLM (Qwen2.5-7B, LoRA), meta-trained on SCM-prior synthetic tasks to write the weights of a small
MLP from k labelled rows per class, transfers to real tables where the stand-in did not: on held-out real
datasets its generated network beats (1) its own fitted ridge scaffold with the generator silenced and
(2) in-context prompting of the same LLM on the same rows.

## Method

- **Generator** (`a3/model.py`). Token sequence: text header with the task shape and the feature names, one
  projected token per labelled row (+ class slot), `" Weights:"`, one token per feature (with its per-class
  means), one per class, one bias token. The last d + C + 1 hidden states go through zero-initialised linear
  heads that write deltas on W1 (one column per feature), b1, and W2/b2 of a d-128-C ReLU MLP.
  W1 starts at [+I; -I], and the second layer is a ridge regression refitted in closed form on every episode's
  support features (R2D2, dual form). With the heads silenced the model is exactly the fitted scaffold.
- **LLM**: Qwen2.5-7B base (bf16, frozen) + LoRA r = 16 on all attention and MLP projections. Scale arms:
  Qwen2.5-1.5B and 0.5B. Control: the 0.5B architecture with random weights, fully trained (does pretraining matter?).
- **Meta-training**: fresh SCM-prior tasks every step (`a3/prior.py`, ported from PR #14: random causal MLPs,
  skewed / rounded / duplicated features, label noise, imbalance), d in 2..40, C in 2..10,
  k in {1, 2, 5, 10, 20}, batch 8 episodes, 20 queries per class, query cross-entropy, 6000 AdamW steps
  (LoRA 1e-4, new layers 5e-4, scaffold 3e-3, warmup then cosine). No real table is ever trained on.
- **Evaluation** (`a3/data.py`): 23 held-out real tables: PR #14's 18-table suite (comparable to the earlier
  numbers) plus 5 OpenML-CC18 tables never used before (credit-g, cmc, pendigits, qsar-biodeg, steel-plates).
  Per table: 5 random 60/40 splits x k in {1, 2, 5, 10, 20} x 4 draws of k support rows per class, balanced
  shuffled queries from the test side. The same episodes for every method.
- **Generator variants reported**: `gen` (raw, the pre-registered primary), `gated` (per episode, use the
  generator only if its leave-one-out ridge accuracy on the support rows beats the scaffold's; a cheap
  "the scaffold is a floor" rule), `gen_anon` (feature names replaced by f1..fd, to measure whether the
  LLM uses the column names).

## Baselines (same k rows, same queries)

| Baseline | Notes |
|---|---|
| Fitted ridge scaffold | generator silenced, with the trained ridge strength (per seed); also untrained (lambda = 0.1) |
| Logistic regression | scikit-learn defaults |
| TabPFN v2 | public classifier checkpoint, default settings |
| In-context prompting, same LLM | Qwen2.5-7B base; rows serialised as text with column names, label letter scored by next-token logit; draw 0 only, 20 queries per episode, prompts up to 7000 tokens (the generator is compared on the same queries) |
| k-NN (k = 5), nearest centroid, majority | references |

## Pass / fail (pre-registered)

Primary metric: query accuracy averaged over episodes per table, then over the 23 tables; 3 training seeds
for the 7B generator; 95 % paired bootstrap CI over tables.

- **PASS** if the 7B generator (`gen`, mean of 3 seeds) beats its own fitted scaffold AND the 7B in-context
  baseline, both with the CI lower bound above 0.
- **FAIL** otherwise. Logistic regression and TabPFN are reported but beating them is not required for PASS.
  If PASS holds only for `gated`, that is reported as a weaker, secondary result.

## Prior work (checked 2026-11-24)

- **MotherNet** (Mueller et al., 2023, [arXiv:2312.08598](https://arxiv.org/abs/2312.08598)) and **HyperFast**
  (Bonet et al., 2024, [arXiv:2402.14335](https://arxiv.org/abs/2402.14335)): transformer hypernetworks trained
  on synthetic tasks (TabPFN-style prior) that write the weights of a tabular MLP from the training set in one
  pass. Closest prior art; neither uses a pretrained LLM or feature names, and neither targets few-shot.
- **TabPFN v2** (Hollmann et al., Nature 2025): in-context tabular foundation model, the strongest public
  small-data baseline.
- **LLM weight generation from text**: NeuroGen ([arXiv:2505.12470](https://arxiv.org/abs/2505.12470)),
  ModelGPT ([arXiv:2402.12408](https://arxiv.org/abs/2402.12408)), IGPG ([arXiv:2504.02012](https://arxiv.org/abs/2504.02012)):
  generate parameters for vision / generic models from prompts; not tested on held-out tabular tasks against
  TabPFN and fitted linear baselines.
- **LLMs on tables via text**: TabLLM (Hegselmann et al., AISTATS 2023) and later in-context work
  (e.g. [arXiv:2502.03147](https://arxiv.org/abs/2502.03147)) find text-serialised LLM ICL competitive at very
  small k and behind numeric models otherwise; that is why the in-context baseline is included.

What is new here: a pretrained LLM body inside a MotherNet-style generator, on top of a fitted closed-form
scaffold that guarantees a classical floor, tested head to head against the same LLM in context and TabPFN.

## Running

`run.sh [--tiny] [--arms a,b]`. Outputs in `$RUN_DIR/out`: one `arm_<name>.json` per arm (per-episode
accuracies), `summary.json` (per-method means, per-k curves, bootstrap comparisons, verdict) and
`fig_accuracy_vs_k.png`. Checkpoints in `$RUN_DIR/ckpt`; finished arms are skipped on restart and an
unfinished training arm resumes. `--tiny` runs 3 tables, 1 draw, Qwen2.5-0.5B for 30 steps.

Arm order (full run): baselines, 0.5B x 3 seeds, 7B seed 0, 7B in-context, 7B seeds 1-2, 1.5B x 3 seeds,
0.5B random-init control. Data is downloaded from OpenML / Hugging Face into `HF_HOME` on first use.
