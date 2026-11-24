# A3: LLM-generated network weights with a real LLM (negative)

**2026-11-24** · code `experiments/a3_llm_weights` · results `results/a3_llm_weights/1288777/` · one 80 GB GPU, ~3 h

## Question
Origin: Random-Ideas PRs #14, #34 and the `llm_weights` project. There, a model was asked to *emit the weights* of a
small network for a tabular task from a short dataset description. Every one of those runs used a CPU stand-in for the
LLM, and the result was flat: a fitted ridge read-out matched logistic regression and the "generator" added nothing.
The real hypothesis — that an actual pretrained LLM, conditioned on a dataset description and a few labelled rows,
emits useful task-specific weights — had never been tested with an LLM. This is that test.

## Setup
A pretrained Qwen2.5 LLM (LoRA-tuned) reads `k` labelled rows per class plus the column names and writes weight deltas
onto a `d`-128-`C` MLP whose second layer is a ridge read-out refitted per episode (the "fitted scaffold"). The model is
meta-trained only on synthetic tasks drawn from an SCM prior, then evaluated frozen on 23 held-out real OpenML tables
(2300 episodes, k = 1..20). Arms: generator at 7B / 1.5B / 0.5B (3 seeds each), a random-init 0.5B control, and a
"gated" variant that can learn to silence the generator and fall back to the scaffold.

Baselines at equal compute / information: the silenced **scaffold** (the same MLP with the generator's deltas zeroed,
read-out still fitted), **logistic regression**, **TabPFN v2**, k-NN, class-centroid, and **in-context prompting** of
the same 7B LLM (icl-7b).

**Pass (pre-stated):** the 7B generator beats *both* its own scaffold *and* 7B in-context prompting, with a 95 %
bootstrap CI over tables above 0.

## Prior work
Hypernetworks (Ha et al., 2017) and parameter-prediction methods generate weights, but from learned encoders, not a
language model reading a prose description. TabPFN (Hollmann et al., 2023; v2 2025) is the strong in-context tabular
learner and is included as a reference ceiling. "Prompt an LLM to output the weights of a net" is the specific claim
from the origin project; no published method was found that does this competitively on real tabular data, which is why
the fitted-scaffold and TabPFN baselines matter.

## Result (mean balanced accuracy over 23 tables, k = 1..20)

| method | accuracy |
|---|---|
| TabPFN v2 (reference ceiling) | 0.699 |
| gen-7b : **scaffold** (generator silenced, read-out fitted) | 0.687 |
| gen-7b : gated (may silence the generator) | 0.687 |
| ridge, untrained scaffold | 0.685 |
| logistic regression | 0.681 |
| **gen-7b : gen (the LLM-written weights)** | **0.668** |
| 7B in-context prompting (icl-7b) | 0.589 |

Paired contrasts for the 7B generator (95 % bootstrap CI over the 23 tables; wins):

| contrast | mean | 95 % CI | wins |
|---|---|---|---|
| gen − scaffold | **−0.018** | [−0.032, −0.005] | 10/23 |
| gen − logreg | −0.013 | [−0.026, −0.001] | 10/23 |
| gen − TabPFN | −0.030 | [−0.054, −0.009] | 10/23 |
| gen@20 − in-context (icl) | +0.076 | [+0.049, +0.107] | 21/23 |
| gated − scaffold | +0.0004 | [−0.001, +0.002] | 8/23 |

Model size does not change it: generator accuracy is 0.668 (7B), 0.673 (1.5B), 0.673 (0.5B) against scaffolds of
0.686–0.687 throughout; the random-init 0.5B generator is 0.658 (its scaffold 0.684). The per-k curves are parallel —
the generator tracks the scaffold a constant ~0.018 below it at every k.

## Verdict
**Fail.** The 7B generator is *significantly worse* than its own fitted scaffold (−0.018, CI excludes 0), so the pass
criterion is not met even though the generator does beat raw in-context prompting (+0.076). The gated variant, free to
use the generator whenever it helps, instead learns to silence it and lands exactly on the scaffold (+0.0004, CI
includes 0). The conclusion from the CPU-scale runs holds with a real LLM up to 7B: **the LLM-written weights add
nothing over the per-episode-fitted linear read-out** — the read-out is doing all the work, and the generated first-layer
deltas are, if anything, mild noise. A larger generator does not help. TabPFN v2 (0.699), a model built for this, stays
ahead of the whole family.

## Figure
`results/a3_llm_weights/1288777/fig_accuracy_vs_k.png` — accuracy vs k per method.

## Why this is a clean negative
The design closes the loopholes of the origin runs: a real pretrained LLM (not a CPU stand-in), three model sizes, a
random-init control to separate "pretrained knowledge" from "capacity", a gated arm that *could* have used the
generator selectively, and held-out real tables rather than synthetic ones. All of them say the same thing. The useful
signal in "LLM-generated weights" for tabular tasks lives entirely in the fitted linear head; the language model's
contribution to the weights is not positive.
