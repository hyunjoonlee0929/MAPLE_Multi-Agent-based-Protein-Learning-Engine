# MAPLE (Multi-Agent Protein Learning Engine)

MAPLE is a multi-agent system that autonomously explores protein sequence space and optimizes enzyme function through iterative learning.

## Core Scope
- Multi-agent loop: planner -> sequence -> structure -> property -> optimization -> evaluation
- Random mutation-based sequence exploration
- Structure adapters: `dummy`, `esmfold`, `alphafold2`
- Embedding + property prediction with uncertainty-aware scoring
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

## Web UI Run (Streamlit)
```bash
cd MAPLE
streamlit run app.py
```

## Real Structure Adapter Integration
MAPLE now supports runtime structure adapter execution for `esmfold` and `alphafold2`.

### ESMFold adapter command (included)
Default command in `config.yaml`:
```bash
python3 scripts/run_esmfold_adapter.py --sequence-file {sequence_file} --output-file {output_file} --allow-mock
```

- If transformers/weights are available: real ESMFold inference result is written.
- If unavailable and `--allow-mock` is set: adapter emits mock JSON and pipeline continues.

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
- optional: `model_id`, `pdb_path`, `pae_mean`, `ptm`, `plddt_mean`, `runtime_sec`, `note`

## Phase 1 Scoring Upgrade
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

### 2) Train NPZ checkpoint
```bash
cd MAPLE
python3 scripts/train_property_numpy.py \
  --data data/sample_property_labels.csv \
  --output checkpoints/property_linear.npz \
  --embedding-dim 128
```

### 3) Run MAPLE with trained checkpoint
```bash
cd MAPLE
python3 main.py \
  --property-checkpoint checkpoints/property_linear.npz \
  --num-iterations 5
```

The Streamlit UI also supports entering the checkpoint path in the sidebar.

## UI Features
- Sidebar controls for runtime/model/scoring parameters
- One-click execution of optimization loop
- Live run summary with best sequence and score
- Iteration trend chart and ranked candidate table
- Downloadable artifacts (`history.json`, `summary.json`)

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

## Phase Roadmap and Transition Gates
MAPLE is now operated with explicit phase gates:
- Current operating phase: **Phase 2** (Adapter-Integrated Optimization)
- Next target phase: **Phase 3** (Model-Driven Design)

Phase 3 transition gates (all required):
- `num_iterations >= 10`
- `best_score >= 0.75`
- `constraint_pass_rate >= 0.70`
- `structure_external_rate >= 0.80`
- `structure_error_fallback_rate <= 0.05`

The run summary now includes `phase_report`, and Streamlit shows unmet gates directly.

### Generate a Phase Gate Report
```bash
cd MAPLE
python3 scripts/phase_gate_report.py --num-iterations 10 --structure-backend esmfold
```

Artifacts:
- `outputs/phase_gate_report/phase_gate_report.json`
- `outputs/phase_gate_report/phase_gate_report.md`
