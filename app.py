"""Streamlit dashboard for MAPLE."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from core.active_learning_jobs import run_active_learning_job
from core.active_learning_view import active_learning_acquisition_rows, active_learning_round_rows
from core.campaign_jobs import run_campaign_job
from core.campaign_view import campaign_acquisition_rows, campaign_round_rows
from core.dbtl_jobs import run_dbtl_ingest_job
from core.dbtl_view import dbtl_summary_row, dbtl_trial_rows
from core.multiobjective import build_pareto_candidate_rows
from core.validation import cv_run_rows, leaderboard_rows
from core.validation_jobs import run_validation_report_jobs
from main import load_config, run_maple


ROOT = Path(__file__).parent
DEFAULT_CONFIG = ROOT / "config.yaml"


st.set_page_config(
    page_title="MAPLE UI",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp {
        background: radial-gradient(circle at 20% 15%, #f5f9f2 0%, #e6f0e5 45%, #d7e7e0 100%);
      }
      .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
      }
      .hero {
        background: linear-gradient(130deg, #0f5132 0%, #1d6f42 55%, #2f855a 100%);
        color: #f8fff9;
        padding: 1.2rem 1.4rem;
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(15, 81, 50, 0.22);
      }
      .hero h1 {
        margin: 0;
        font-size: 1.6rem;
        letter-spacing: 0.02em;
      }
      .hero p {
        margin: 0.35rem 0 0 0;
        opacity: 0.95;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>MAPLE: Multi-Agent Protein Learning Engine</h1>
      <p>Configure the in-silico optimization loop, run experiments, and inspect sequence evolution artifacts in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

cfg = load_config(DEFAULT_CONFIG)



def _safe_float(value, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _quick_profile_defaults(name: str) -> dict:
    profiles = {
        "fast_demo": {
            "num_iterations": 3,
            "num_candidates": 8,
            "top_k": 3,
            "mutation_rate": 1,
            "selection_strategy": "diverse",
            "scoring_preset": "balanced",
            "structure_backend": "esmfold",
            "structure_strict": False,
            "constraint_enabled": False,
        },
        "balanced_research": {
            "num_iterations": 8,
            "num_candidates": 16,
            "top_k": 4,
            "mutation_rate": 2,
            "selection_strategy": "pareto",
            "scoring_preset": "balanced",
            "structure_backend": "esmfold",
            "structure_strict": False,
            "constraint_enabled": True,
        },
        "structure_priority": {
            "num_iterations": 8,
            "num_candidates": 16,
            "top_k": 4,
            "mutation_rate": 2,
            "selection_strategy": "pareto_bo",
            "scoring_preset": "structure_first",
            "structure_backend": "esmfold",
            "structure_strict": True,
            "constraint_enabled": True,
        },
    }
    return dict(profiles.get(name, profiles["balanced_research"]))


def _load_json_if_exists(path_text: str) -> dict | None:
    candidate = Path(path_text).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if not candidate.exists():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _render_validation_reports(leaderboard_path_text: str, cv_report_path_text: str) -> None:
    st.subheader("Property Validation Reports")

    leaderboard_payload = _load_json_if_exists(leaderboard_path_text)
    cv_payload = _load_json_if_exists(cv_report_path_text)

    if leaderboard_payload is None and cv_payload is None:
        st.info("Validation report files not found yet. Run retraining/validation scripts first.")
        return

    if leaderboard_payload is not None:
        st.caption("Checkpoint validation leaderboard")
        lb_rows = leaderboard_rows(leaderboard_payload)
        lb_df = pd.DataFrame(lb_rows)
        if not lb_df.empty:
            c1, c2 = st.columns(2)
            c1.metric("Best Checkpoint", str(lb_df.iloc[0]["checkpoint"]))
            c2.metric("Best Val RMSE", f"{float(lb_df.iloc[0]['val_rmse_mean']):.4f}")
            st.dataframe(lb_df, use_container_width=True)

    if cv_payload is not None:
        st.caption("Cross-seed reproducibility")
        summary = cv_payload.get("summary", {})
        rmse_summary = summary.get("val_mean_rmse", {})
        c1, c2 = st.columns(2)
        c1.metric("CV Val RMSE Mean", f"{_safe_float(rmse_summary.get('mean'), 0.0):.4f}")
        c2.metric("CV Val RMSE Std", f"{_safe_float(rmse_summary.get('std'), 0.0):.4f}")

        cv_df = pd.DataFrame(cv_run_rows(cv_payload))
        if not cv_df.empty:
            st.line_chart(cv_df.set_index("split_seed")[["val_rmse_mean", "val_mae_mean"]])
            st.dataframe(cv_df, use_container_width=True)


def _render_active_learning_report(active_learning_report_path_text: str) -> None:
    st.subheader("Active Learning Reports")
    payload = _load_json_if_exists(active_learning_report_path_text)
    if payload is None:
        st.info("Active learning report not found yet. Run active learning cycle first.")
        return

    round_df = pd.DataFrame(active_learning_round_rows(payload))
    if not round_df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rounds", int(len(round_df)))
        c2.metric("Final Train Size", int(round_df.iloc[-1]["train_size"]))
        c3.metric("Latest Val RMSE", f"{float(round_df.iloc[-1]['val_rmse_mean']):.4f}")

        st.caption("Round-wise validation and acquisition trend")
        st.line_chart(round_df.set_index("round")[["val_rmse_mean", "train_rmse_mean"]])
        st.line_chart(round_df.set_index("round")[["acq_mean", "pseudo_stability_mean", "pseudo_activity_mean"]])
        st.dataframe(round_df, use_container_width=True)

    acq_df = pd.DataFrame(active_learning_acquisition_rows(payload))
    if not acq_df.empty:
        st.caption("Acquired batch details")
        st.dataframe(acq_df, use_container_width=True)


def _render_campaign_report(campaign_report_path_text: str) -> None:
    st.subheader("Closed-Loop Campaign Reports")
    payload = _load_json_if_exists(campaign_report_path_text)
    if payload is None:
        st.info("Campaign report not found yet. Run closed-loop campaign first.")
        return

    round_df = pd.DataFrame(campaign_round_rows(payload))
    if not round_df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Campaign Rounds", int(len(round_df)))
        c2.metric("Latest MAPLE Best", f"{float(round_df.iloc[-1]['maple_best_score']):.4f}")
        c3.metric("Latest Val RMSE", f"{float(round_df.iloc[-1]['val_rmse_mean']):.4f}")

        st.line_chart(round_df.set_index("round")[["maple_best_score", "val_rmse_mean", "train_rmse_mean"]])
        st.line_chart(round_df.set_index("round")[["acquired_stability_mean", "acquired_activity_mean"]])
        st.dataframe(round_df, use_container_width=True)

    acq_df = pd.DataFrame(campaign_acquisition_rows(payload))
    if not acq_df.empty:
        st.caption("Campaign acquired sequences")
        st.dataframe(acq_df, use_container_width=True)


def _render_dbtl_report(dbtl_report_path_text: str) -> None:
    st.subheader("DBTL Ingestion Reports")
    payload = _load_json_if_exists(dbtl_report_path_text)
    if payload is None:
        st.info("DBTL report not found yet. Run DBTL ingestion first.")
        return

    summary = dbtl_summary_row(payload)
    c1, c2, c3 = st.columns(3)
    c1.metric("Imported Records", int(summary["imported_records"]))
    c2.metric("Retrain Triggered", "Yes" if bool(summary["retrain_triggered"]) else "No")
    c3.metric("Val RMSE (Retrained)", f"{float(summary['val_rmse_mean']):.4f}")

    st.dataframe(pd.DataFrame([summary]), use_container_width=True)

    trial_df = pd.DataFrame(dbtl_trial_rows(payload))
    if not trial_df.empty:
        st.caption("Retrain hyperparameter trials")
        st.line_chart(trial_df.set_index("ridge_alpha")[["val_mean_rmse", "val_mean_mae"]])
        st.dataframe(trial_df, use_container_width=True)


with st.sidebar:
    st.header("Run Controls")
    ui_mode = st.radio(
        "Parameter Mode",
        options=["Simple", "Advanced"],
        index=0,
        help="Simple mode exposes only core controls. Advanced mode exposes all tunables.",
    )
    profile_name = st.selectbox(
        "Quick Profile",
        options=["fast_demo", "balanced_research", "structure_priority"],
        index=1,
        help="Load recommended starting defaults for common goals.",
    )

    runtime = dict(cfg.get("runtime", {}))
    model = dict(cfg.get("model", {}))
    profile = _quick_profile_defaults(profile_name)
    defaults = {
        "num_iterations": int(profile.get("num_iterations", cfg.get("num_iterations", 5))),
        "num_candidates": int(profile.get("num_candidates", runtime.get("num_candidates", 10))),
        "top_k": int(profile.get("top_k", runtime.get("top_k", 3))),
        "mutation_rate": int(profile.get("mutation_rate", runtime.get("mutation_rate", 1))),
        "selection_strategy": str(profile.get("selection_strategy", runtime.get("selection_strategy", "diverse"))),
        "scoring_preset": str(profile.get("scoring_preset", runtime.get("scoring_preset", "balanced"))),
        "structure_backend": str(profile.get("structure_backend", model.get("structure_backend", "esmfold"))),
        "structure_strict": bool(profile.get("structure_strict", model.get("structure_strict", False))),
        "constraint_enabled": bool(profile.get("constraint_enabled", runtime.get("constraint_enabled", False))),
        "embedding_backend": str(model.get("embedding_backend", "random")),
    }

    with st.expander("Parameter Guide", expanded=False):
        st.markdown(
            """
            - `num_candidates`: each iteration candidate pool size.
            - `top_k`: elite sequences retained before mutation.
            - `mutation_rate`: number of residue substitutions per generated child.
            - `selection_strategy`: `diverse`, `elitist`, `pareto`, `pareto_bo`.
            - `structure_backend`: `esmfold` recommended for real structure path.
            - `structure_strict`: if enabled, fail run when external structure call fails.
            - `embedding_backend`: choose `esm2`/`prott5` for real PLM embeddings, `random` for fast MVP.
            - `constraint_enabled`: enforce feasibility thresholds before elite selection.
            """
        )

    seed_sequence = st.text_input(
        "Seed Sequence",
        value=cfg.get("seed_sequence", ""),
        help="Initial protein sequence used to start iterative optimization.",
    )
    seed = st.number_input(
        "Global Seed",
        min_value=0,
        value=int(cfg.get("seed", 42)),
        step=1,
        help="Controls reproducibility for sequence generation and ranking.",
    )
    num_iterations = st.slider(
        "Iterations",
        min_value=1,
        max_value=100,
        value=defaults["num_iterations"],
        help="Number of planner->...->evaluation loop cycles.",
    )

    st.subheader("Candidate Generation")
    num_candidates = st.slider(
        "Candidates / Iter",
        min_value=2,
        max_value=100,
        value=defaults["num_candidates"],
        help="How many candidate sequences are evaluated per iteration.",
    )
    top_k = st.slider(
        "Top-K Elites",
        min_value=1,
        max_value=20,
        value=defaults["top_k"],
        help="Top-ranked parents used to generate next iteration.",
    )
    mutation_rate = st.slider(
        "Mutation Rate",
        min_value=1,
        max_value=10,
        value=defaults["mutation_rate"],
        help="Number of amino acid edits per mutation step.",
    )
    selection_strategy = st.selectbox(
        "Selection Strategy",
        options=["diverse", "elitist", "pareto", "pareto_bo"],
        index=["diverse", "elitist", "pareto", "pareto_bo"].index(
            defaults["selection_strategy"]
            if defaults["selection_strategy"] in {"diverse", "elitist", "pareto", "pareto_bo"}
            else "diverse"
        ),
        help="`pareto`/`pareto_bo` are preferred for multi-objective optimization.",
    )
    bo_beta = st.slider(
        "BO Beta (pareto_bo)",
        min_value=0.0,
        max_value=2.0,
        value=_safe_float(runtime.get("bo_beta", 0.30), 0.30),
        step=0.05,
        disabled=selection_strategy != "pareto_bo",
        help="Higher values increase novelty/exploration in BO acquisition.",
    )
    bo_trials_per_parent = st.slider(
        "BO Trials / Parent",
        min_value=1,
        max_value=32,
        value=int(runtime.get("bo_trials_per_parent", 8)),
        step=1,
        disabled=selection_strategy != "pareto_bo",
        help="Mutation trials generated per parent before acquisition ranking.",
    )
    scoring_preset = st.selectbox(
        "Scoring Preset",
        options=["balanced", "exploration", "structure_first", "activity_first"],
        index=["balanced", "exploration", "structure_first", "activity_first"].index(
            defaults["scoring_preset"]
            if defaults["scoring_preset"] in {"balanced", "exploration", "structure_first", "activity_first"}
            else "balanced"
        ),
        help="Applies predefined score-weight priorities.",
    )
    use_weight_preset = st.checkbox(
        "Use Weight Preset",
        value=bool(runtime.get("use_weight_preset", True)),
        help="When enabled, selected preset can overwrite raw score weights.",
    )
    normalize_score_weights = st.checkbox(
        "Normalize Score Weights",
        value=bool(runtime.get("normalize_score_weights", True)),
        help="Rescales all weights so total equals 1.",
    )
    min_hamming_distance = st.slider(
        "Min Hamming Distance",
        min_value=0,
        max_value=10,
        value=int(runtime.get("min_hamming_distance", 2)),
        help="Diversity threshold between selected elites.",
    )

    constraint_enabled = defaults["constraint_enabled"]
    if ui_mode == "Advanced":
        constraint_enabled = st.checkbox(
            "Enable Constraints",
            value=bool(defaults["constraint_enabled"]),
            help="Filter or penalize candidates that fail feasibility thresholds.",
        )
    constraint_mode = st.selectbox(
        "Constraint Mode",
        options=["hard", "soft"],
        index=0 if str(runtime.get("constraint_mode", "hard")).lower() == "hard" else 1,
        disabled=(ui_mode != "Advanced") or (not constraint_enabled),
        help="`hard`: remove infeasible; `soft`: keep with score penalty.",
    )
    constraint_penalty = st.slider(
        "Constraint Penalty (soft mode)",
        min_value=0.0,
        max_value=2.0,
        value=_safe_float(runtime.get("constraint_penalty", 0.20), 0.20),
        step=0.01,
        disabled=(ui_mode != "Advanced") or (not constraint_enabled) or constraint_mode != "soft",
    )
    constraint_enabled_effective = bool(constraint_enabled) if ui_mode == "Advanced" else False
    min_stability = st.slider(
        "Min Stability",
        min_value=-5.0,
        max_value=5.0,
        value=_safe_float(runtime.get("min_stability", -5.0), -5.0),
        step=0.05,
        disabled=(ui_mode != "Advanced") or (not constraint_enabled_effective),
    )
    min_activity = st.slider(
        "Min Activity",
        min_value=-5.0,
        max_value=5.0,
        value=_safe_float(runtime.get("min_activity", -5.0), -5.0),
        step=0.05,
        disabled=(ui_mode != "Advanced") or (not constraint_enabled_effective),
    )
    min_structure_confidence = st.slider(
        "Min Structure Confidence",
        min_value=0.0,
        max_value=1.0,
        value=_safe_float(runtime.get("min_structure_confidence", 0.0), 0.0),
        step=0.01,
        disabled=(ui_mode != "Advanced") or (not constraint_enabled_effective),
    )
    min_plddt = st.slider(
        "Min pLDDT",
        min_value=0.0,
        max_value=100.0,
        value=_safe_float(runtime.get("min_plddt", 0.0), 0.0),
        step=1.0,
        disabled=(ui_mode != "Advanced") or (not constraint_enabled_effective),
    )
    min_ptm = st.slider(
        "Min pTM",
        min_value=0.0,
        max_value=1.0,
        value=_safe_float(runtime.get("min_ptm", 0.0), 0.0),
        step=0.01,
        disabled=(ui_mode != "Advanced") or (not constraint_enabled_effective),
    )
    max_pae = st.slider(
        "Max PAE",
        min_value=0.0,
        max_value=50.0,
        value=_safe_float(runtime.get("max_pae", 50.0), 50.0),
        step=0.5,
        disabled=(ui_mode != "Advanced") or (not constraint_enabled_effective),
    )

    st.subheader("Scoring Weights")
    w_stability = st.slider(
        "Stability Weight",
        min_value=0.0,
        max_value=1.0,
        value=_safe_float(runtime.get("w_stability", 0.40), 0.40),
        step=0.01,
        disabled=ui_mode != "Advanced",
    )
    w_activity = st.slider(
        "Activity Weight",
        min_value=0.0,
        max_value=1.0,
        value=_safe_float(runtime.get("w_activity", 0.40), 0.40),
        step=0.01,
        disabled=ui_mode != "Advanced",
    )
    w_uncertainty = st.slider(
        "Uncertainty Weight",
        min_value=0.0,
        max_value=1.0,
        value=_safe_float(runtime.get("w_uncertainty", 0.10), 0.10),
        step=0.01,
        disabled=ui_mode != "Advanced",
    )
    w_structure = st.slider(
        "Structure Weight",
        min_value=0.0,
        max_value=1.0,
        value=_safe_float(runtime.get("w_structure", 0.10), 0.10),
        step=0.01,
        disabled=ui_mode != "Advanced",
    )
    w_plddt = st.slider(
        "pLDDT Weight",
        min_value=0.0,
        max_value=1.0,
        value=_safe_float(runtime.get("w_plddt", 0.05), 0.05),
        step=0.01,
        disabled=ui_mode != "Advanced",
    )
    w_ptm = st.slider(
        "pTM Weight",
        min_value=0.0,
        max_value=1.0,
        value=_safe_float(runtime.get("w_ptm", 0.03), 0.03),
        step=0.01,
        disabled=ui_mode != "Advanced",
    )
    w_pae = st.slider(
        "PAE(inv) Weight",
        min_value=0.0,
        max_value=1.0,
        value=_safe_float(runtime.get("w_pae", 0.02), 0.02),
        step=0.01,
        disabled=ui_mode != "Advanced",
    )

    st.subheader("Model")
    structure_backend = st.selectbox(
        "Structure Backend",
        options=["dummy", "esmfold", "alphafold2"],
        index=["dummy", "esmfold", "alphafold2"].index(
            defaults["structure_backend"] if defaults["structure_backend"] in {"dummy", "esmfold", "alphafold2"} else "esmfold"
        ),
        help="Use `esmfold` for real adapter integration path.",
    )
    structure_strict = st.checkbox(
        "Strict Structure Runtime",
        value=bool(defaults["structure_strict"]),
        disabled=ui_mode != "Advanced",
        help="When enabled, adapter failures stop the run instead of silent mock fallback.",
    )
    esmfold_command = st.text_input(
        "ESMFold External Command (Optional)",
        value=str(model.get("esmfold_command", "")),
        help="Use placeholders {sequence_file} and {output_file}. If empty, adapter mock is used.",
    )
    alphafold2_command = st.text_input(
        "AlphaFold2 External Command (Optional)",
        value=str(model.get("alphafold2_command", "")),
        help="Use placeholders {sequence_file} and {output_file}. If empty, adapter mock is used.",
    )
    structure_timeout_sec = st.slider(
        "Structure Timeout (sec)",
        min_value=5,
        max_value=600,
        value=int(model.get("structure_timeout_sec", 120)),
        step=5,
        disabled=ui_mode != "Advanced",
    )
    structure_retries = st.slider(
        "Structure Retries",
        min_value=0,
        max_value=5,
        value=int(model.get("structure_retries", 1)),
        step=1,
        disabled=ui_mode != "Advanced",
    )
    structure_batch_size = st.slider(
        "Structure Batch Size",
        min_value=1,
        max_value=128,
        value=int(model.get("structure_batch_size", 16)),
        step=1,
        disabled=ui_mode != "Advanced",
    )
    embedding_dim = st.slider(
        "Embedding Dim",
        min_value=8,
        max_value=1024,
        value=int(model.get("embedding_dim", 128)),
        step=8,
        disabled=ui_mode != "Advanced",
    )
    embedding_backend = st.selectbox(
        "Embedding Backend",
        options=["random", "esm2", "prott5"],
        index=["random", "esm2", "prott5"].index(
            defaults["embedding_backend"] if defaults["embedding_backend"] in {"random", "esm2", "prott5"} else "random"
        ),
        help="Protein embedding backend for property prediction.",
        disabled=ui_mode != "Advanced",
    )
    default_embedding_model_id = str(model.get("embedding_model_id", "") or "")
    embedding_model_id = st.text_input(
        "Embedding Model ID (optional)",
        value=default_embedding_model_id,
        help="Leave empty to use backend default model id.",
        disabled=ui_mode != "Advanced",
    )
    embedding_device = st.selectbox(
        "Embedding Device",
        options=["cpu", "auto", "cuda"],
        index=["cpu", "auto", "cuda"].index(
            str(model.get("embedding_device", "cpu"))
            if str(model.get("embedding_device", "cpu")) in {"cpu", "auto", "cuda"}
            else "cpu"
        ),
        disabled=ui_mode != "Advanced",
    )
    embedding_pooling = st.selectbox(
        "Embedding Pooling",
        options=["mean", "cls"],
        index=0 if str(model.get("embedding_pooling", "mean")).lower() != "cls" else 1,
        disabled=ui_mode != "Advanced",
    )
    embedding_allow_mock = st.checkbox(
        "Allow Embedding Mock Fallback",
        value=bool(model.get("embedding_allow_mock", True)),
        help="When enabled, falls back to random embedding if model load/inference fails.",
        disabled=ui_mode != "Advanced",
    )
    uncertainty_samples = st.slider(
        "Uncertainty Samples",
        min_value=1,
        max_value=20,
        value=int(model.get("uncertainty_samples", 5)),
        disabled=ui_mode != "Advanced",
    )
    uncertainty_noise = st.slider(
        "Uncertainty Noise",
        min_value=0.0,
        max_value=0.2,
        value=_safe_float(model.get("uncertainty_noise", 0.02), 0.02),
        step=0.005,
        disabled=ui_mode != "Advanced",
    )
    property_checkpoint = st.text_input(
        "Property Checkpoint (.npz or .pt)",
        value=str(model.get("property_checkpoint", "")),
        help="Optional path to trained property model checkpoint.",
    )

    st.subheader("Validation Report Paths")
    leaderboard_report_path = st.text_input(
        "Leaderboard JSON",
        value="outputs/property_validation/validation_leaderboard.json",
        help="Path to validation_leaderboard.json",
    )
    cv_report_path = st.text_input(
        "CV Report JSON",
        value="outputs/property_cv/property_cv_report.json",
        help="Path to property_cv_report.json",
    )
    validation_data_path = st.text_input(
        "Validation Data CSV",
        value="data/sample_property_labels.csv",
        help="Dataset path used by validation scripts.",
    )
    validation_checkpoints_csv = st.text_input(
        "Validation Checkpoints (.npz/.pt, comma-separated)",
        value="checkpoints/property_linear.npz",
    )
    validation_val_ratio = st.slider(
        "Validation Ratio",
        min_value=0.05,
        max_value=0.5,
        value=0.2,
        step=0.05,
    )
    validation_split_seed = st.number_input("Validation Split Seed", min_value=0, value=42, step=1)
    validation_split_mode = st.selectbox("Validation Split Mode", options=["random", "scaffold"], index=0)
    validation_scaffold_k = st.slider("Validation Scaffold K", min_value=1, max_value=6, value=3, step=1)
    validation_ensemble_size = st.slider("Validation Ensemble Size", min_value=1, max_value=8, value=1, step=1)
    validation_ece_bins = st.slider("Validation ECE Bins", min_value=2, max_value=30, value=10, step=1)
    validation_split_seeds_csv = st.text_input("CV Split Seeds", value="1,7,13,21,42")
    validation_ridge_alphas_csv = st.text_input("CV Ridge Alphas", value="1e-4,1e-3,1e-2,1e-1")
    validation_leaderboard_output_dir = st.text_input(
        "Leaderboard Output Dir",
        value="outputs/property_validation",
    )
    validation_cv_output_dir = st.text_input(
        "CV Report Output Dir",
        value="outputs/property_cv",
    )
    st.subheader("Active Learning")
    al_data_path = st.text_input("AL Data CSV", value="data/sample_property_labels.csv")
    al_output_dir = st.text_input("AL Output Dir", value="outputs/active_learning")
    al_report_path = st.text_input(
        "AL Report JSON",
        value="outputs/active_learning/active_learning_report.json",
    )
    al_checkpoint_out = st.text_input(
        "AL Checkpoint Out",
        value="checkpoints/property_linear_active_learning.npz",
    )
    al_rounds = st.slider("AL Rounds", min_value=1, max_value=10, value=3, step=1)
    al_batch_size = st.slider("AL Batch Size", min_value=1, max_value=20, value=4, step=1)
    al_pool_size = st.slider("AL Pool Size", min_value=10, max_value=200, value=40, step=5)
    al_beta = st.slider("AL Beta", min_value=0.0, max_value=2.0, value=0.30, step=0.05)
    al_split_seed = st.number_input("AL Split Seed", min_value=0, value=42, step=1)
    al_ridge_alphas = st.text_input("AL Ridge Alphas", value="1e-4,1e-3,1e-2,1e-1")
    st.subheader("Closed-Loop Campaign")
    campaign_report_path = st.text_input(
        "Campaign Report JSON",
        value="outputs/closed_loop_campaign/campaign_report.json",
    )
    campaign_config_path = st.text_input("Campaign Config", value="config.yaml")
    campaign_data_path = st.text_input("Campaign Data CSV", value="data/sample_property_labels.csv")
    campaign_output_dir = st.text_input("Campaign Output Dir", value="outputs/closed_loop_campaign")
    campaign_rounds = st.slider("Campaign Rounds", min_value=1, max_value=10, value=3, step=1)
    campaign_maple_iterations = st.slider("Campaign MAPLE Iterations", min_value=1, max_value=20, value=3, step=1)
    campaign_acquisition_batch_size = st.slider("Campaign Acquisition Batch", min_value=1, max_value=20, value=4, step=1)
    st.subheader("DBTL Ingestion")
    dbtl_report_path = st.text_input(
        "DBTL Report JSON",
        value="outputs/dbtl_ingest/dbtl_retrain_report.json",
    )
    dbtl_seed_data = st.text_input("DBTL Seed Data CSV", value="data/sample_property_labels.csv")
    dbtl_input_path = st.text_input("DBTL Input File", value="data/sample_dbtl_results.csv")
    dbtl_format = st.selectbox("DBTL Input Format", options=["auto", "csv", "json"], index=0)
    dbtl_output_dir = st.text_input("DBTL Output Dir", value="outputs/dbtl_ingest")
    dbtl_checkpoint_out = st.text_input("DBTL Checkpoint Out", value="checkpoints/property_linear_dbtl.npz")
    dbtl_min_records = st.slider("DBTL Min Records to Retrain", min_value=1, max_value=100, value=1, step=1)
    run_dbtl_ingest_clicked = st.button(
        "Run DBTL Ingestion + Retrain",
        use_container_width=True,
    )
    run_campaign_clicked = st.button(
        "Run Closed-Loop Campaign",
        use_container_width=True,
    )
    run_active_learning_clicked = st.button(
        "Run Active Learning Cycle",
        use_container_width=True,
    )
    generate_validation_reports_clicked = st.button(
        "Generate Validation Reports",
        use_container_width=True,
    )

    st.caption(
        "Simple mode runs with profile defaults and hides advanced research knobs. "
        "Switch to Advanced to edit full constraints/weights/model settings."
    )
    run_clicked = st.button("Run MAPLE", type="primary", use_container_width=True)

if run_active_learning_clicked:
    with st.spinner("Running active learning cycle..."):
        al_result = run_active_learning_job(
            root=ROOT,
            data_path=al_data_path.strip(),
            output_dir=al_output_dir.strip(),
            checkpoint_out=al_checkpoint_out.strip(),
            embedding_dim=int(embedding_dim),
            embedding_backend=str(embedding_backend),
            embedding_model_id=str(embedding_model_id or ""),
            embedding_device=str(embedding_device),
            embedding_pooling=str(embedding_pooling),
            embedding_allow_mock=bool(embedding_allow_mock),
            val_ratio=float(validation_val_ratio),
            split_seed=int(al_split_seed),
            split_mode=str(validation_split_mode),
            scaffold_k=int(validation_scaffold_k),
            ensemble_size=int(validation_ensemble_size),
            ece_bins=int(validation_ece_bins),
            rounds=int(al_rounds),
            batch_size=int(al_batch_size),
            pool_size=int(al_pool_size),
            mutation_rate=int(mutation_rate),
            beta=float(al_beta),
            ridge_alphas=al_ridge_alphas.strip(),
            seed=int(seed),
        )
    if al_result.ok:
        st.success("Active learning cycle completed.")
    else:
        st.error("Active learning cycle failed. Check logs below.")
    with st.expander(f"Active Learning Job (rc={al_result.returncode})", expanded=not al_result.ok):
        st.code(" ".join(al_result.command))
        if al_result.stdout.strip():
            st.text(al_result.stdout.strip())
        if al_result.stderr.strip():
            st.text(al_result.stderr.strip())

if run_campaign_clicked:
    with st.spinner("Running closed-loop campaign..."):
        campaign_result = run_campaign_job(
            root=ROOT,
            config_path=campaign_config_path.strip(),
            data_path=campaign_data_path.strip(),
            output_dir=campaign_output_dir.strip(),
            rounds=int(campaign_rounds),
            maple_iterations=int(campaign_maple_iterations),
            acquisition_batch_size=int(campaign_acquisition_batch_size),
            embedding_dim=int(embedding_dim),
            embedding_backend=str(embedding_backend),
            embedding_model_id=str(embedding_model_id or ""),
            embedding_device=str(embedding_device),
            embedding_pooling=str(embedding_pooling),
            embedding_allow_mock=bool(embedding_allow_mock),
            val_ratio=float(validation_val_ratio),
            split_seed=int(validation_split_seed),
            split_mode=str(validation_split_mode),
            scaffold_k=int(validation_scaffold_k),
            ensemble_size=int(validation_ensemble_size),
            ece_bins=int(validation_ece_bins),
            ridge_alphas=validation_ridge_alphas_csv.strip(),
            selection_strategy=selection_strategy,
            bo_beta=float(bo_beta),
            bo_trials_per_parent=int(bo_trials_per_parent),
            num_candidates=int(num_candidates),
            top_k=int(top_k),
            mutation_rate=int(mutation_rate),
            seed=int(seed),
        )
    if campaign_result.ok:
        st.success("Closed-loop campaign completed.")
    else:
        st.error("Closed-loop campaign failed. Check logs below.")
    with st.expander(f"Campaign Job (rc={campaign_result.returncode})", expanded=not campaign_result.ok):
        st.code(" ".join(campaign_result.command))
        if campaign_result.stdout.strip():
            st.text(campaign_result.stdout.strip())
        if campaign_result.stderr.strip():
            st.text(campaign_result.stderr.strip())

if run_dbtl_ingest_clicked:
    with st.spinner("Running DBTL ingestion and retrain trigger..."):
        dbtl_result = run_dbtl_ingest_job(
            root=ROOT,
            seed_data=dbtl_seed_data.strip(),
            dbtl_input=dbtl_input_path.strip(),
            dbtl_format=dbtl_format,
            output_dir=dbtl_output_dir.strip(),
            checkpoint_out=dbtl_checkpoint_out.strip(),
            embedding_dim=int(embedding_dim),
            embedding_backend=str(embedding_backend),
            embedding_model_id=str(embedding_model_id or ""),
            embedding_device=str(embedding_device),
            embedding_pooling=str(embedding_pooling),
            embedding_allow_mock=bool(embedding_allow_mock),
            val_ratio=float(validation_val_ratio),
            split_seed=int(validation_split_seed),
            split_mode=str(validation_split_mode),
            scaffold_k=int(validation_scaffold_k),
            ensemble_size=int(validation_ensemble_size),
            ece_bins=int(validation_ece_bins),
            ridge_alphas=validation_ridge_alphas_csv.strip(),
            min_imported_records=int(dbtl_min_records),
        )
    if dbtl_result.ok:
        st.success("DBTL ingestion completed.")
    else:
        st.error("DBTL ingestion failed. Check logs below.")
    with st.expander(f"DBTL Job (rc={dbtl_result.returncode})", expanded=not dbtl_result.ok):
        st.code(" ".join(dbtl_result.command))
        if dbtl_result.stdout.strip():
            st.text(dbtl_result.stdout.strip())
        if dbtl_result.stderr.strip():
            st.text(dbtl_result.stderr.strip())

if generate_validation_reports_clicked:
    with st.spinner("Generating validation leaderboard and CV report..."):
        job_results = run_validation_report_jobs(
            root=ROOT,
            data_path=validation_data_path.strip(),
            checkpoints_csv=validation_checkpoints_csv.strip(),
            val_ratio=float(validation_val_ratio),
            split_seed=int(validation_split_seed),
            split_mode=str(validation_split_mode),
            scaffold_k=int(validation_scaffold_k),
            ensemble_size=int(validation_ensemble_size),
            ece_bins=int(validation_ece_bins),
            split_seeds_csv=validation_split_seeds_csv.strip(),
            ridge_alphas_csv=validation_ridge_alphas_csv.strip(),
            leaderboard_output_dir=validation_leaderboard_output_dir.strip(),
            cv_output_dir=validation_cv_output_dir.strip(),
        )

    if job_results and all(item.ok for item in job_results):
        st.success("Validation reports generated successfully.")
    else:
        st.error("Validation report generation failed. Check command logs below.")

    for item in job_results:
        with st.expander(f"Job: {item.name} (rc={item.returncode})", expanded=not item.ok):
            st.code(" ".join(item.command))
            if item.stdout.strip():
                st.text(item.stdout.strip())
            if item.stderr.strip():
                st.text(item.stderr.strip())

if run_clicked:
    overrides = {
        "seed": int(seed),
        "seed_sequence": seed_sequence.strip(),
        "num_iterations": int(num_iterations),
        "num_candidates": int(num_candidates),
        "top_k": int(top_k),
        "mutation_rate": int(mutation_rate),
        "selection_strategy": selection_strategy,
        "bo_beta": float(bo_beta),
        "bo_trials_per_parent": int(bo_trials_per_parent),
        "scoring_preset": scoring_preset,
        "use_weight_preset": bool(use_weight_preset),
        "normalize_score_weights": bool(normalize_score_weights),
        "min_hamming_distance": int(min_hamming_distance),
        "constraint_enabled": bool(constraint_enabled_effective),
        "constraint_mode": constraint_mode,
        "constraint_penalty": None if not constraint_enabled_effective else float(constraint_penalty),
        "min_stability": None if not constraint_enabled_effective else float(min_stability),
        "min_activity": None if not constraint_enabled_effective else float(min_activity),
        "min_structure_confidence": None if not constraint_enabled_effective else float(min_structure_confidence),
        "min_plddt": None if not constraint_enabled_effective else float(min_plddt),
        "min_ptm": None if not constraint_enabled_effective else float(min_ptm),
        "max_pae": None if not constraint_enabled_effective else float(max_pae),
        "w_stability": float(w_stability),
        "w_activity": float(w_activity),
        "w_uncertainty": float(w_uncertainty),
        "w_structure": float(w_structure),
        "w_plddt": float(w_plddt),
        "w_ptm": float(w_ptm),
        "w_pae": float(w_pae),
        "structure_backend": structure_backend,
        "esmfold_command": esmfold_command.strip() or None,
        "alphafold2_command": alphafold2_command.strip() or None,
        "structure_timeout_sec": int(structure_timeout_sec),
        "structure_retries": int(structure_retries),
        "structure_strict": bool(structure_strict),
        "structure_batch_size": int(structure_batch_size),
        "embedding_dim": int(embedding_dim),
        "embedding_backend": str(embedding_backend),
        "embedding_model_id": embedding_model_id.strip() or None,
        "embedding_device": str(embedding_device),
        "embedding_pooling": str(embedding_pooling),
        "embedding_allow_mock": bool(embedding_allow_mock),
        "property_checkpoint": property_checkpoint.strip() or None,
        "uncertainty_samples": int(uncertainty_samples),
        "uncertainty_noise": float(uncertainty_noise),
        "validation_leaderboard_path": leaderboard_report_path.strip(),
        "validation_cv_report_path": cv_report_path.strip(),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "outputs" / f"ui_run_{timestamp}"

    with st.spinner("Running multi-agent optimization loop..."):
        final_state, resolved, artifact_dir = run_maple(
            config=cfg,
            overrides=overrides,
            output_dir=output_dir,
        )

    history = final_state.get("history", [])
    best_seq = final_state["sequences"][0] if final_state.get("sequences") else None
    best_score = float(final_state["scores"][0]) if final_state.get("scores") else None

    structure_mode = None
    if final_state.get("structures"):
        structure_mode = final_state["structures"][0].get("mode")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Iterations", resolved["num_iterations"])
    col2.metric("Best Score", f"{best_score:.4f}" if best_score is not None else "N/A")
    col3.metric("Candidates", len(final_state.get("sequences", [])))
    col4.metric("Structure Mode", structure_mode or "N/A")

    if structure_mode == "mock" and structure_backend in {"esmfold", "alphafold2"}:
        st.warning(
            f"{structure_backend} adapter is running in mock mode. "
            "Provide a valid external command to switch to external mode."
        )

    constraint_summary = final_state.get("constraint_summary", {})
    if constraint_summary.get("enabled"):
        pass_rate_pct = 100.0 * float(constraint_summary.get("passed", 0)) / max(1, int(constraint_summary.get("total", 0)))
        c1, c2 = st.columns(2)
        c1.metric("Constraint Pass Rate", f"{pass_rate_pct:.1f}%")
        c2.metric(
            "Constraint Passed",
            f"{constraint_summary.get('passed', 0)}/{constraint_summary.get('total', 0)}",
        )
        st.info(
            "Constrained optimization enabled: "
            f"{constraint_summary.get('passed', 0)}/{constraint_summary.get('total', 0)} "
            "candidates satisfied constraints."
        )

    st.subheader("Top Sequence")
    st.code(best_seq or "N/A")

    st.subheader("History")
    hist_df = pd.DataFrame(history)
    if not hist_df.empty:
        st.line_chart(hist_df.set_index("iteration")[["best_score", "mean_score"]])
        if "constraint_pass_rate" in hist_df.columns:
            st.caption("Constraint pass rate trend")
            st.line_chart(hist_df.set_index("iteration")[["constraint_pass_rate"]])

        if {"structure_external_rate", "structure_mock_rate", "structure_error_fallback_rate"}.issubset(hist_df.columns):
            st.caption("Structure adapter mode rates")
            st.line_chart(
                hist_df.set_index("iteration")[[
                    "structure_external_rate",
                    "structure_mock_rate",
                    "structure_error_fallback_rate",
                ]]
            )
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("No history records were produced.")

    st.subheader("Top Ranked Candidates")
    records = []
    for i, seq in enumerate(final_state.get("sequences", [])[:20]):
        prop = final_state.get("properties", [])[i]
        structure = final_state.get("structures", [])[i] if i < len(final_state.get("structures", [])) else {}
        records.append(
            {
                "rank": i + 1,
                "sequence": seq,
                "score": final_state.get("scores", [None])[i],
                "stability": prop.get("stability"),
                "activity": prop.get("activity"),
                "uncertainty": prop.get("uncertainty"),
                "structure_backend": structure.get("backend"),
                "structure_mode": structure.get("mode"),
                "structure_confidence": structure.get("confidence"),
                "plddt_mean": structure.get("plddt_mean"),
                "ptm": structure.get("ptm"),
                "pae_mean": structure.get("pae_mean"),
            }
        )
    st.dataframe(pd.DataFrame(records), use_container_width=True)

    st.subheader("Multi-Objective Pareto View")
    pareto_rows = build_pareto_candidate_rows(final_state)
    pareto_df = pd.DataFrame(pareto_rows)
    if not pareto_df.empty:
        p1, p2 = st.columns(2)
        front_count = int(pareto_df["is_pareto_front"].sum())
        p1.metric("Pareto Front Size", front_count)
        p2.metric("Total Candidates", int(len(pareto_df)))

        st.caption("Stability vs Activity (all candidates)")
        st.scatter_chart(pareto_df, x="stability", y="activity")

        front_df = pareto_df[pareto_df["is_pareto_front"]]
        if not front_df.empty:
            st.caption("Pareto front only")
            st.scatter_chart(front_df, x="stability", y="activity")

        st.dataframe(
            pareto_df[
                [
                    "rank",
                    "sequence",
                    "score",
                    "stability",
                    "activity",
                    "uncertainty",
                    "structure_confidence",
                    "pareto_rank",
                    "is_pareto_front",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.info("Pareto view unavailable for empty candidate list.")

    st.subheader("Artifacts")
    st.write(f"Saved to: `{artifact_dir}`")
    history_json = artifact_dir / "history.json"
    summary_json = artifact_dir / "summary.json"

    if history_json.exists():
        st.download_button(
            "Download history.json",
            data=history_json.read_text(encoding="utf-8"),
            file_name="history.json",
            mime="application/json",
            use_container_width=True,
        )

    if summary_json.exists():
        st.download_button(
            "Download summary.json",
            data=summary_json.read_text(encoding="utf-8"),
            file_name="summary.json",
            mime="application/json",
            use_container_width=True,
        )

    st.subheader("Resolved Settings")
    st.code(json.dumps(resolved, indent=2))
elif not generate_validation_reports_clicked and not run_active_learning_clicked and not run_campaign_clicked and not run_dbtl_ingest_clicked:
    st.info("Configure parameters in the sidebar, then click 'Run MAPLE'.")

st.divider()
_render_validation_reports(
    leaderboard_path_text=leaderboard_report_path,
    cv_report_path_text=cv_report_path,
)
st.divider()
_render_active_learning_report(active_learning_report_path_text=al_report_path)
st.divider()
_render_campaign_report(campaign_report_path_text=campaign_report_path)
st.divider()
_render_dbtl_report(dbtl_report_path_text=dbtl_report_path)
