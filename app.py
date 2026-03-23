"""Streamlit dashboard for MAPLE."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

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

with st.sidebar:
    st.header("Run Controls")
    seed_sequence = st.text_input("Seed Sequence", value=cfg.get("seed_sequence", ""))
    seed = st.number_input("Global Seed", min_value=0, value=int(cfg.get("seed", 42)), step=1)
    num_iterations = st.slider("Iterations", min_value=1, max_value=100, value=int(cfg.get("num_iterations", 5)))

    st.subheader("Candidate Generation")
    runtime = dict(cfg.get("runtime", {}))
    num_candidates = st.slider("Candidates / Iter", min_value=2, max_value=100, value=int(runtime.get("num_candidates", 10)))
    top_k = st.slider("Top-K Elites", min_value=1, max_value=20, value=int(runtime.get("top_k", 3)))
    mutation_rate = st.slider("Mutation Rate", min_value=1, max_value=10, value=int(runtime.get("mutation_rate", 1)))
    selection_strategy = st.selectbox(
        "Selection Strategy",
        options=["diverse", "elitist"],
        index=0 if runtime.get("selection_strategy", "diverse") == "diverse" else 1,
    )
    min_hamming_distance = st.slider(
        "Min Hamming Distance",
        min_value=0,
        max_value=10,
        value=int(runtime.get("min_hamming_distance", 2)),
    )
    constraint_enabled = st.checkbox("Enable Constraints", value=bool(runtime.get("constraint_enabled", False)))
    min_stability = st.slider("Min Stability", min_value=-5.0, max_value=5.0, value=float(runtime.get("min_stability", -5.0)), step=0.05)
    min_activity = st.slider("Min Activity", min_value=-5.0, max_value=5.0, value=float(runtime.get("min_activity", -5.0)), step=0.05)
    min_structure_confidence = st.slider(
        "Min Structure Confidence",
        min_value=0.0,
        max_value=1.0,
        value=float(runtime.get("min_structure_confidence", 0.0)),
        step=0.01,
    )
    min_plddt = st.slider("Min pLDDT", min_value=0.0, max_value=100.0, value=float(runtime.get("min_plddt", 0.0)), step=1.0)
    min_ptm = st.slider("Min pTM", min_value=0.0, max_value=1.0, value=float(runtime.get("min_ptm", 0.0)), step=0.01)
    max_pae = st.slider("Max PAE", min_value=0.0, max_value=50.0, value=float(runtime.get("max_pae", 50.0)), step=0.5)

    st.subheader("Scoring Weights")
    w_stability = st.slider("Stability Weight", min_value=0.0, max_value=1.0, value=float(runtime.get("w_stability", 0.40)), step=0.01)
    w_activity = st.slider("Activity Weight", min_value=0.0, max_value=1.0, value=float(runtime.get("w_activity", 0.40)), step=0.01)
    w_uncertainty = st.slider("Uncertainty Weight", min_value=0.0, max_value=1.0, value=float(runtime.get("w_uncertainty", 0.10)), step=0.01)
    w_structure = st.slider("Structure Weight", min_value=0.0, max_value=1.0, value=float(runtime.get("w_structure", 0.10)), step=0.01)
    w_plddt = st.slider("pLDDT Weight", min_value=0.0, max_value=1.0, value=float(runtime.get("w_plddt", 0.05)), step=0.01)
    w_ptm = st.slider("pTM Weight", min_value=0.0, max_value=1.0, value=float(runtime.get("w_ptm", 0.03)), step=0.01)
    w_pae = st.slider("PAE(inv) Weight", min_value=0.0, max_value=1.0, value=float(runtime.get("w_pae", 0.02)), step=0.01)

    st.subheader("Model")
    model = dict(cfg.get("model", {}))
    structure_backend = st.selectbox(
        "Structure Backend",
        options=["dummy", "esmfold", "alphafold2"],
        index=["dummy", "esmfold", "alphafold2"].index(model.get("structure_backend", "dummy")),
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
    )
    structure_retries = st.slider(
        "Structure Retries",
        min_value=0,
        max_value=5,
        value=int(model.get("structure_retries", 1)),
        step=1,
    )
    structure_batch_size = st.slider(
        "Structure Batch Size",
        min_value=1,
        max_value=128,
        value=int(model.get("structure_batch_size", 16)),
        step=1,
    )
    embedding_dim = st.slider("Embedding Dim", min_value=8, max_value=1024, value=int(model.get("embedding_dim", 128)), step=8)
    uncertainty_samples = st.slider("Uncertainty Samples", min_value=1, max_value=20, value=int(model.get("uncertainty_samples", 5)))
    uncertainty_noise = st.slider("Uncertainty Noise", min_value=0.0, max_value=0.2, value=float(model.get("uncertainty_noise", 0.02)), step=0.005)
    property_checkpoint = st.text_input(
        "Property Checkpoint (.npz or .pt)",
        value=str(model.get("property_checkpoint", "")),
        help="Optional path to trained property model checkpoint.",
    )

    run_clicked = st.button("Run MAPLE", type="primary", use_container_width=True)

if run_clicked:
    overrides = {
        "seed": int(seed),
        "seed_sequence": seed_sequence.strip(),
        "num_iterations": int(num_iterations),
        "num_candidates": int(num_candidates),
        "top_k": int(top_k),
        "mutation_rate": int(mutation_rate),
        "selection_strategy": selection_strategy,
        "min_hamming_distance": int(min_hamming_distance),
        "constraint_enabled": bool(constraint_enabled),
        "min_stability": None if not constraint_enabled else float(min_stability),
        "min_activity": None if not constraint_enabled else float(min_activity),
        "min_structure_confidence": None if not constraint_enabled else float(min_structure_confidence),
        "min_plddt": None if not constraint_enabled else float(min_plddt),
        "min_ptm": None if not constraint_enabled else float(min_ptm),
        "max_pae": None if not constraint_enabled else float(max_pae),
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
        "structure_batch_size": int(structure_batch_size),
        "embedding_dim": int(embedding_dim),
        "property_checkpoint": property_checkpoint.strip() or None,
        "uncertainty_samples": int(uncertainty_samples),
        "uncertainty_noise": float(uncertainty_noise),
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
else:
    st.info("Configure parameters in the sidebar, then click 'Run MAPLE'.")
