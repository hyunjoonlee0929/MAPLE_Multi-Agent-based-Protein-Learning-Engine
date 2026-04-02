# MAPLE (Multi-Agent based Protein Learning Engine)

MAPLE is a multi-agent system that autonomously explores protein sequence space and optimizes enzyme function through iterative learning.

## Core Scope
- Multi-agent loop: planner -> sequence -> structure -> property -> optimization -> evaluation
- Random mutation-based sequence exploration
- Structure adapters: `dummy`, `esmfold`, `alphafold2`
- Embedding backends: `random`, `esm2`, `prott5` (with optional mock fallback)
- Property prediction with uncertainty-aware scoring
- Structure-confidence-aware scoring (`w_structure`)
- Diversity-aware evolutionary optimization
- Iteration artifact export (JSON/CSV)

## Install
```bash
cd MAPLE
python3 -m pip install -r requirements.txt
```

## CLI Run
```bash
cd MAPLE
python3 main.py --num-iterations 5 --selection-strategy diverse --min-hamming-distance 2
```

### Embedding Backend Run (ESM2 / ProtT5)
```bash
cd MAPLE
python3 main.py \
  --embedding-backend esm2 \
  --embedding-model-id facebook/esm2_t12_35M_UR50D \
  --embedding-device auto \
  --embedding-pooling mean
```

Disable random fallback for strict embedding runtime:
```bash
python3 main.py --embedding-backend esm2 --disable-embedding-mock-fallback
```

## Web UI Run (Streamlit)
```bash
cd MAPLE
streamlit run app.py
```

## Real Structure Adapter Integration
MAPLE now supports runtime structure adapter execution for `esmfold` and `alphafold2`.
Default runtime path in `config.yaml` is now `structure_backend: "esmfold"` with adapter fallback enabled.

### ESMFold adapter command (included)
Default command in `config.yaml`:
```bash
python3 scripts/run_esmfold_adapter.py --sequence-file {sequence_file} --output-file {output_file} --allow-mock
```

- If transformers/weights are available: real ESMFold inference result is written.
- If unavailable and `--allow-mock` is set: adapter emits mock JSON and pipeline continues.
- To enforce strict runtime behavior, set `model.structure_strict: true` (or CLI `--structure-strict`) so adapter failures stop the run.

### Run with ESMFold backend
```bash
cd MAPLE
python3 main.py \
  --structure-backend esmfold \
  --esmfold-command "python3 scripts/run_esmfold_adapter.py --sequence-file {sequence_file} --output-file {output_file} --allow-mock" \
  --structure-timeout-sec 120 \
  --structure-retries 1 \
  --structure-batch-size 16
```

### External command contract
Your external backend command must:
- read sequence from `{sequence_file}`
- write JSON to `{output_file}`

Recommended JSON keys:
- `confidence` (float)
- `engine` (string)
- standardized output fields: `plddt_mean`, `ptm`, `pae_mean`, `pdb_path`
- optional: `model_id`, `runtime_sec`, `note`

## Scoring Upgrade
Evaluation score now combines:
- stability (`w_stability`)
- activity (`w_activity`)
- uncertainty (`w_uncertainty`)
- structure confidence (`w_structure`)
- pLDDT (`w_plddt`)
- pTM (`w_ptm`)
- inverse PAE (`w_pae`)

Default weights in `config.yaml`:
- `w_stability: 0.35`
- `w_activity: 0.35`
- `w_uncertainty: 0.10`
- `w_structure: 0.10`
- `w_plddt: 0.05`
- `w_ptm: 0.03`
- `w_pae: 0.02`

## Constrained Optimization
Optimization can enforce feasibility constraints before selecting elites.

Runtime constraints:
- `constraint_enabled`
- `constraint_mode` (`hard` or `soft`)
- `constraint_penalty` (used in `soft` mode)
- `min_stability`
- `min_activity`
- `min_structure_confidence`
- `min_plddt`
- `min_ptm`
- `max_pae`

Example:
```bash
cd MAPLE
python3 main.py \
  --constraint-enabled \
  --min-plddt 60 \
  --min-ptm 0.5 \
  --max-pae 20 \
  --min-stability 0.2 \
  --min-activity 0.2
```

In `soft` mode, violating candidates are not dropped, but their scores are penalized.

MAPLE now records per-iteration constraint tracking in history:
- `constraint_pass_rate`
- `constraint_passed`
- `constraint_total`
- `constraint_mode`

## Multi-Objective Optimization (Pareto + BO)
MAPLE now supports multi-objective sequence selection strategies:
- `selection_strategy=pareto`
- `selection_strategy=pareto_bo`

Pareto objectives (maximization):
- stability
- activity
- negative uncertainty
- structure confidence

`pareto_bo` uses a lightweight linear surrogate + novelty-based uncertainty to rank mutated candidates with:
- acquisition = `predicted_score + bo_beta * novelty_uncertainty`

Relevant runtime params:
- `bo_beta`
- `bo_trials_per_parent`

CLI example:
```bash
cd MAPLE
python3 main.py \
  --selection-strategy pareto_bo \
  --bo-beta 0.35 \
  --bo-trials-per-parent 10 \
  --num-iterations 5
```

## Scoring Presets
You can use preset profiles and automatic weight normalization:
- `balanced`
- `exploration`
- `structure_first`
- `activity_first`

CLI examples:
```bash
cd MAPLE
python3 main.py --scoring-preset structure_first
python3 main.py --scoring-preset exploration --disable-score-weight-normalization
```

## Upgrade Step: Labeled Property Model
You can train a lightweight labeled surrogate model (stability/activity) and plug it into MAPLE.

### 1) Prepare labeled CSV
Expected columns:
- `sequence`
- `stability`
- `activity`

A sample dataset is included at:
- `data/sample_property_labels.csv`

### 2) Train NPZ checkpoint with validation metrics
```bash
cd MAPLE
python3 scripts/train_property_numpy.py \
  --data data/sample_property_labels.csv \
  --output checkpoints/property_linear.npz \
  --embedding-dim 128 \
  --embedding-backend esm2 \
  --embedding-model-id facebook/esm2_t12_35M_UR50D \
  --val-ratio 0.2 \
  --split-seed 42 \
  --metrics-out outputs/property_metrics/property_train_metrics.json
```

The training pipeline now performs:
- train/validation split (`--split-mode random|scaffold`)
- ridge-regression fitting on train split
- optional bootstrap ensemble (`--ensemble-size > 1`)
- validation metric reporting (`RMSE`, `MAE`, `R2`, `Pearson`) for stability/activity and mean
- uncertainty calibration summary (`val_calibration.ece`)
- metrics artifact export as JSON

### 2-1) Retraining Pipeline (validation-driven model selection)
```bash
cd MAPLE
python3 scripts/retrain_property_pipeline.py \
  --data data/sample_property_labels.csv \
  --embedding-backend esm2 \
  --split-mode scaffold \
  --ensemble-size 5 \
  --ridge-alphas "1e-4,1e-3,1e-2,1e-1" \
  --checkpoint-out checkpoints/property_linear_best.npz \
  --output-dir outputs/property_retrain
```

Outputs:
- `checkpoints/property_linear_best.npz`
- `outputs/property_retrain/retrain_report.json`

### 2-1a) Create fixed validation split (reusable)
```bash
cd MAPLE
python3 scripts/make_validation_split.py \
  --data data/sample_property_labels.csv \
  --val-ratio 0.2 \
  --split-seed 42 \
  --output outputs/property_validation/fixed_val_split.json
```

Then use the same validation split for retraining:
```bash
cd MAPLE
python3 scripts/retrain_property_pipeline.py \
  --data data/sample_property_labels.csv \
  --ridge-alphas "1e-4,1e-3,1e-2,1e-1" \
  --val-index-file outputs/property_validation/fixed_val_split.json \
  --checkpoint-out checkpoints/property_linear_best.npz \
  --output-dir outputs/property_retrain
```

### 2-2) Checkpoint Validation Leaderboard
```bash
cd MAPLE
python3 scripts/evaluate_property_checkpoints.py \
  --data data/sample_property_labels.csv \
  --checkpoints checkpoints/property_linear.npz,checkpoints/property_linear_best.npz \
  --val-ratio 0.2 \
  --split-seed 42 \
  --output-dir outputs/property_validation
```

Outputs:
- `outputs/property_validation/validation_leaderboard.json`
- `outputs/property_validation/validation_leaderboard.csv`

### 2-3) Cross-seed reproducibility report
```bash
cd MAPLE
python3 scripts/property_cv_report.py \
  --data data/sample_property_labels.csv \
  --split-seeds "1,7,13,21,42" \
  --ridge-alphas "1e-4,1e-3,1e-2,1e-1" \
  --output-dir outputs/property_cv
```

Output:
- `outputs/property_cv/property_cv_report.json`

## Active Learning Cycle
You can run iterative active learning with pseudo-label acquisition and retraining:

```bash
cd MAPLE
python3 scripts/active_learning_cycle.py \
  --data data/sample_property_labels.csv \
  --rounds 3 \
  --batch-size 4 \
  --pool-size 40 \
  --beta 0.30 \
  --checkpoint-out checkpoints/property_linear_active_learning.npz \
  --output-dir outputs/active_learning
```

Outputs:
- `outputs/active_learning/active_learning_report.json`
- `outputs/active_learning/augmented_dataset.csv`
- `checkpoints/property_linear_active_learning.npz`

## Closed-Loop In-Silico Campaign
This campaign script ties together:
- MAPLE design loop (`pareto` / `pareto_bo`)
- pseudo-label acquisition
- round-wise property model retraining

```bash
cd MAPLE
python3 scripts/closed_loop_campaign.py \
  --config config.yaml \
  --data data/sample_property_labels.csv \
  --rounds 3 \
  --maple-iterations 3 \
  --acquisition-batch-size 4 \
  --selection-strategy pareto_bo \
  --bo-beta 0.30 \
  --output-dir outputs/closed_loop_campaign
```

Outputs:
- `outputs/closed_loop_campaign/campaign_report.json`
- `outputs/closed_loop_campaign/train_dataset_final.csv`
- per-round model checkpoints and MAPLE artifacts

## Public Dataset Benchmark Pipeline
Use a JSON manifest of public datasets and automatically generate reproducible benchmark tables.

### 1) Prepare manifest
Template file:
- `benchmarks/public_datasets_manifest.json`

Each dataset entry should include:
- `name`
- `path` (CSV with `sequence,stability,activity`)
- `source_url`
- `license`

### 2) Run benchmark
```bash
cd MAPLE
python3 scripts/benchmark_public_datasets.py \
  --manifest benchmarks/public_datasets_manifest.json \
  --output-dir outputs/public_benchmark \
  --embedding-backends random \
  --split-mode scaffold \
  --split-seeds "1,7,13,21,42" \
  --ensemble-size 3
```

Outputs:
- `outputs/public_benchmark/public_benchmark_report.json`
- `outputs/public_benchmark/public_benchmark_leaderboard.csv`
- `outputs/public_benchmark/public_benchmark_table.md`

## DBTL Ingestion and Auto-Retrain Trigger
DBTL test records can be ingested to automatically refresh the property model.

DBTL record schema:
- `sequence` (required)
- `stability` (required)
- `activity` (required)
- optional: `experiment_id`, `split` (`train|val`), `source`, `timestamp`, `assay`

Reference schema file:
- `docs/dbtl_record_schema.json`

Example DBTL CSV:
- `data/sample_dbtl_results.csv`

Run ingestion + retrain trigger:
```bash
cd MAPLE
python3 scripts/dbtl_ingest_retrain.py \
  --seed-data data/sample_property_labels.csv \
  --dbtl-input data/sample_dbtl_results.csv \
  --dbtl-format auto \
  --output-dir outputs/dbtl_ingest \
  --checkpoint-out checkpoints/property_linear_dbtl.npz
```

Outputs:
- `outputs/dbtl_ingest/dbtl_retrain_report.json`
- `outputs/dbtl_ingest/train_dataset_merged.csv`
- `outputs/dbtl_ingest/val_dataset_merged.csv`
- `checkpoints/property_linear_dbtl.npz` (when trigger condition is met)

### 3) Run MAPLE with trained checkpoint
```bash
cd MAPLE
python3 main.py \
  --property-checkpoint checkpoints/property_linear.npz \
  --num-iterations 5
```

The Streamlit UI also supports entering the checkpoint path in the sidebar.

## UI Features
- `Simple` / `Advanced` parameter mode switch in sidebar
- `Quick Profile` presets (`fast_demo`, `balanced_research`, `structure_priority`)
- Inline parameter guide for key optimization/model knobs
- Sidebar controls for runtime/model/scoring parameters
- One-click execution of optimization loop
- Live run summary with best sequence and score
- Iteration trend chart and ranked candidate table
- Validation report panel for:
  - checkpoint leaderboard (`validation_leaderboard.json`)
  - cross-seed reproducibility (`property_cv_report.json`)
- Sidebar button to generate validation reports directly from UI
- Pareto front view for current run candidates (stability/activity/objective rank)
- Sidebar button to run active learning cycle and inspect logs
- Active learning report panel (round-wise val metrics and acquired-batch trend)
- Closed-loop campaign panel (round-wise MAPLE score / retrain metrics / acquired sequences)
- DBTL ingestion panel (auto-retrain trigger logs and retrain metric summary)
- Downloadable artifacts (`history.json`, `summary.json`)

`summary.json` now includes `validation_reports` when report paths are provided.
`history.json`/`history.csv` include per-iteration tracking fields:
- `validation_linked`
- `validation_best_checkpoint`
- `validation_best_val_rmse`
- `validation_cv_rmse_mean`
- `validation_cv_rmse_std`

## Key Files
- `main.py`: reusable run service + CLI entrypoint
- `app.py`: Streamlit UI dashboard
- `scripts/run_esmfold_adapter.py`: real/mock ESMFold adapter command
- `scripts/train_property_numpy.py`: label-based NPZ property model trainer
- `core/pipeline.py`: orchestration loop
- `core/reporting.py`: artifact export
- `agents/*.py`: independently testable agent logic
- `tests/*.py`: unit/integration tests

## Constraint Mode Auto-Comparison
You can automatically compare `hard` vs `soft` constraint strategies under the same seed/settings.

```bash
cd MAPLE
python3 scripts/compare_constraint_modes.py \
  --num-iterations 3 \
  --structure-backend dummy \
  --min-plddt 60 \
  --max-pae 20 \
  --constraint-penalty 0.2
```

Generated artifacts:
- `outputs/constraint_compare/constraint_comparison.json`
- `outputs/constraint_compare/constraint_comparison.md`
- full run artifacts under `outputs/constraint_compare/hard` and `outputs/constraint_compare/soft`

## Structure Adapter Monitoring
Per-iteration adapter observability fields are now stored in `history`:
- `structure_external_rate`
- `structure_mock_rate`
- `structure_error_fallback_rate`
- `structure_external`, `structure_mock`, `structure_error_fallback`, `structure_total`

These metrics are visualized in the Streamlit UI as a trend chart.
