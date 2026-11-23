# Random AI Missions

Follow-up experiments for hypotheses from [Random-Ideas](https://github.com/Dannynis/Random-Ideas),
run at a scale the original daily runs could not reach: large models, LLMs, more seeds,
real datasets, and long training runs.

## Layout

| Path | What it holds |
|---|---|
| `HYPOTHESES.md` | Backlog of hypotheses, ranked, each with its origin and what a strong test needs |
| `LOG.md` | Dated lab notebook: what was launched, what came back, what it means |
| `experiments/<name>/` | Code for one experiment: `run.sh` entry point, Python, `README.md` with hypothesis and protocol |
| `results/<name>/<run>/` | Small outputs of each run: `metrics.json`, figures |
| `reports/<date>-<name>/` | Write-ups with figures once an experiment has a verdict |

## How a run works

Each experiment has a `run.sh` entry point. It reads two environment variables: `RUN_DIR/out` for small
results (metrics, figures) and `RUN_DIR/ckpt` for checkpoints. Every `run.sh` resumes from its checkpoint
if one exists, so interrupted runs can be restarted. Results are committed with a `LOG.md` entry.

## Rules for results

- State the hypothesis and the pass/fail criterion before the run.
- At least 3 seeds for any claim; report mean and spread.
- Always include the strongest simple baseline at equal compute.
- Negative results are written up the same way as positive ones.
