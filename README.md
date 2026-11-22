# Reasoning Distillation: ReasonDistill vs Logit KD

Distilling a reasoning teacher (Qwen3.5-4B) into a 2B student (Qwen3.5-2B) on AM-DeepSeek R1 traces, comparing plain
logit distillation (**LKD**) with **ReasonDistill (RD)**: LKD plus a loss that makes a student layer predict the
teacher's reasoning features, read from a sparse autoencoder (SAE) on the teacher. A second arm repeats the study with
Llama-3.2-1B students of DeepSeek-R1-Distill-Llama-8B.

The original Qwen SAE turned out to be dead (its RD targets were ~0), so the SAE was retrained with normalised input and
RD was retrained with it ("new-SAE RD"). That run is the RD of the paper; the earlier "October RD" appears in the text only.

## Start here
| | |
|---|---|
| `results/paper_tables.md` | every table of the paper, as numbers, with the file each came from |
| `results/README.md` | each paper section -> summary files -> the configs, scripts and raw outputs behind them |
| `docs/rd_newsae.md` | the SAE fix and new-SAE RD, including the pre-registered tests |
| `docs/RUN_LOG.md` | chronological lab log of every run and decision (section numbers are cited throughout) |
| `docs/REPO_MAP.md` | naming scheme, old -> new file names, what was archived |

## Layout
```
configs/        train/  eval/  mech_interp/       every config used for a reported run (configs/README.md)
scripts/        train_* eval_* sae_* mi_* analyze_* run_*   flat, prefix = role (scripts/README.md)
custom_eval/    evaluation harness (Qwen): benchmarks, prompts, generation, scoring, pass@k
llama_eval/     the Llama arm's evaluation harness (batched_evals)
training/ losses/ models/ data_loaders/ utils/   training code (upstream)
tests/          unit tests: python -m unittest discover -s tests
docs/           RUN_LOG, new-SAE notes, harness notes, repo map
results/        paper tables and the results index
eval_outputs/   summaries (tracked) and raw per-question outputs (not tracked)
checkpoints/ logs/ cache/   local only (gitignored)
```

## Setup
- Main environment, Python 3.12: `pip install -r requirements-lock.txt`, the exact package set that produced every
  reported result (torch 2.9.1, transformers 5.8.0). `requirements.txt` / `requirements-eval.txt` are the upstream pins and
  are newer; results may shift slightly under them. CUDA GPU for training and generation; analyses run on CPU. Launchers use `PY` (default `python`): `export PY=/path/to/venv/bin/python`.
- Long-context runs (harder benchmarks, ProcessBench, SAE training, MI): a second environment with
  flash-linear-attention 0.5.2 + fla-core 0.5.2 + einops 0.8.2 on top of the main one (RUN_LOG 8q); launchers use
  `PYF` (default `venv_fla/bin/python`).
- Machine-specific locations are settings with this lab machine's paths as defaults: `LLAMA_KD_DIR` (Llama students),
  `KD_BASE_DIR` (Llama teacher SAE + ReasonScore), `RD_TRAIN_REPO` (see below), and the usual `HF_HOME` / `HF_HUB_CACHE`.

## Models and checkpoints
- Hub, `Khaquan/qwen-khaquanS-distillations`: LKD (`qwen-logitKD-final/qwen_logit_kd_continue/epoch_1`), October RD
  (`qwen-reasondistill-october/qwen_reasondistill_final_continue/epoch_1`), the original SAE
  (`qwen-3.5-4B-L16_16x-SAE`).
- Local (`checkpoints/`, not in git; a copy of the RD checkpoints is on the lab OneDrive, `RD_newSAE_checkpoints`):
  new SAE `qwen_sae_norm64/full_l1_003/final/`, RD's copy of it `rd_newsae_sae/`, new-SAE RD students
  `rd_newsae_{low,moderate,high}/final_step_94/`, their start point `rd_newsae_student_init/`, probe tensors `rd_newsae_probe/`.
- New-SAE RD was trained with the RD code at commit `a25867d` (branch `awesomeReasonDistill`), not this branch's trainer.
  To retrain or rerun the head scripts: `git worktree add ../rd_train_a25867d a25867d`, then run that checkout's
  `scripts/train.py` with `configs/train/rd_newsae_*.yaml` from this repo's root (or set `RD_TRAIN_REPO`).

## Reproduce
```bash
bash scripts/run_all_analyses.sh          # every reported summary from the raw outputs (CPU, ~6 min)
bash scripts/train_logit_kd.sh            # LKD (GPU)
bash scripts/run_sae_norm64_full.sh       # retrained SAE + health checks (GPU)
bash scripts/run_rd_newsae_suite.sh       # full evaluation suite for new-SAE RD moderate (GPU)
python -m custom_eval.run_eval --config configs/eval/<config>.yaml   # any single evaluation
```
The raw outputs that the analyses read (`eval_outputs/**/*.json`, about 6 GB) are not in git; they are on the lab
machine. Without them, the tracked summaries are the record.

## Not in this repo
Experiments that are not in the paper (the first RD run's results, the alpha sweep, prompt experiments, old helper
scripts, smoke runs) were moved to `../archive_paper_cleanup_2026-11-22/` on the lab machine; nothing was deleted.
Error-recovery experiments were removed from the study earlier. Lists: `docs/REPO_MAP.md`.
