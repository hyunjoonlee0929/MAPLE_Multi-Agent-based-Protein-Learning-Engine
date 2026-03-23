from __future__ import annotations

import numpy as np
import pytest

from agents.evaluation_agent import EvaluationAgent
from agents.optimization_agent import OptimizationAgent
from agents.planner import PlannerAgent
from agents.property_agent import PropertyAgent
from agents.sequence_agent import SequenceAgent
from agents.structure_agent import StructureAgent
from core.state import create_initial_state
from models.structure_model import build_structure_predictor
from utils.diversity import hamming_distance



def test_planner_agent_sets_defaults() -> None:
    state = create_initial_state("MKTFFV")
    state = PlannerAgent(default_num_candidates=6, default_top_k=2).run(state)
    assert state["config"]["num_candidates"] == 6
    assert state["config"]["top_k"] == 2
    assert state["config"]["mutation_rate"] == 1
    assert state["config"]["w_structure"] == 0.10
    assert state["config"]["w_plddt"] == 0.05
    assert state["config"]["w_ptm"] == 0.03
    assert state["config"]["w_pae"] == 0.02
    assert state["config"]["constraint_enabled"] is False



def test_planner_agent_applies_preset_and_normalizes_weights() -> None:
    state = create_initial_state("MKTFFV")
    state["config"] = {
        "scoring_preset": "activity_first",
        "use_weight_preset": True,
        "normalize_score_weights": True,
    }
    updated = PlannerAgent().run(state)

    keys = ["w_stability", "w_activity", "w_uncertainty", "w_structure", "w_plddt", "w_ptm", "w_pae"]
    total = sum(float(updated["config"][k]) for k in keys)
    assert abs(total - 1.0) < 1e-8
    assert updated["config"]["w_activity"] > updated["config"]["w_stability"]



def test_sequence_agent_generates_configured_candidate_count() -> None:
    state = create_initial_state("MKTFFV")
    state["config"] = {"num_candidates": 5, "mutation_rate": 1}
    updated = SequenceAgent(random_seed=10).run(state)

    assert len(updated["sequences"]) == 5
    assert updated["structures"] == []
    assert updated["properties"] == []
    assert updated["next_sequences"] is None



def test_structure_agent_outputs_one_structure_per_sequence() -> None:
    state = create_initial_state("MKTFFV")
    state["sequences"] = ["MKTFFV", "MKTFFI"]
    updated = StructureAgent().run(state)

    assert len(updated["structures"]) == 2
    assert updated["structures"][0]["backend"] == "dummy_structure_predictor"



def test_structure_predictor_builder_for_dummy_backend() -> None:
    predictor = build_structure_predictor("dummy")
    out = predictor.predict("MKTFFV")
    assert out["backend"] == "dummy_structure_predictor"
    assert out["mode"] == "mock"



def test_structure_predictor_builder_for_esmfold_backend() -> None:
    predictor = build_structure_predictor("esmfold")
    out = predictor.predict("MKTFFV")
    assert out["backend"] == "esmfold_adapter"
    assert out["mode"] in {"mock", "external"}



def test_structure_predictor_builder_for_alphafold2_backend() -> None:
    predictor = build_structure_predictor("alphafold2")
    out = predictor.predict("MKTFFV")
    assert out["backend"] == "alphafold2_adapter"
    assert out["mode"] in {"mock", "external"}



def test_structure_predictor_builder_rejects_invalid_backend() -> None:
    with pytest.raises(ValueError):
        build_structure_predictor("unknown_backend")



def test_property_agent_generates_embeddings_and_properties() -> None:
    state = create_initial_state("MKTFFV")
    state["sequences"] = ["MKTFFV", "MKTFFI"]
    updated = PropertyAgent(embedding_dim=16).run(state)

    assert len(updated["embeddings"]) == 2
    assert updated["embeddings"][0].shape == (16,)
    assert len(updated["properties"]) == 2
    assert "stability" in updated["properties"][0]
    assert "activity" in updated["properties"][0]
    assert "uncertainty" in updated["properties"][0]



def test_evaluation_agent_filters_invalid_and_ranks_scores() -> None:
    state = create_initial_state("MKTFFV")
    state["iteration"] = 1
    state["config"] = {
        "w_stability": 0.35,
        "w_activity": 0.35,
        "w_uncertainty": 0.10,
        "w_structure": 0.10,
        "w_plddt": 0.05,
        "w_ptm": 0.03,
        "w_pae": 0.02,
    }
    state["sequences"] = ["MKTFFV", "INVALIDX"]
    state["structures"] = [{"confidence": 0.7, "plddt_mean": 70, "ptm": 0.7, "pae_mean": 8}, {"confidence": 0.1}]
    state["embeddings"] = [np.ones((4,), dtype=np.float32), np.zeros((4,), dtype=np.float32)]
    state["properties"] = [
        {"stability": 0.8, "activity": 0.9, "uncertainty": 0.2},
        {"stability": 0.1, "activity": 0.2, "uncertainty": 0.1},
    ]

    updated = EvaluationAgent().run(state)

    assert updated["sequences"] == ["MKTFFV"]
    assert len(updated["scores"]) == 1
    assert updated["history"][-1]["iteration"] == 1



def test_evaluation_agent_uncertainty_weight_can_change_ranking() -> None:
    state = create_initial_state("AAAA")
    state["config"] = {
        "w_stability": 0.0,
        "w_activity": 0.0,
        "w_uncertainty": 1.0,
        "w_structure": 0.0,
        "w_plddt": 0.0,
        "w_ptm": 0.0,
        "w_pae": 0.0,
    }
    state["sequences"] = ["AAAA", "AAAT"]
    state["structures"] = [{"confidence": 0.1}, {"confidence": 0.1}]
    state["embeddings"] = [np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)]
    state["properties"] = [
        {"stability": 0.0, "activity": 0.0, "uncertainty": 0.1},
        {"stability": 0.0, "activity": 0.0, "uncertainty": 0.9},
    ]

    updated = EvaluationAgent().run(state)
    assert updated["sequences"][0] == "AAAT"



def test_evaluation_agent_structure_weight_can_change_ranking() -> None:
    state = create_initial_state("AAAA")
    state["config"] = {
        "w_stability": 0.0,
        "w_activity": 0.0,
        "w_uncertainty": 0.0,
        "w_structure": 1.0,
        "w_plddt": 0.0,
        "w_ptm": 0.0,
        "w_pae": 0.0,
    }
    state["sequences"] = ["AAAA", "AAAT"]
    state["structures"] = [{"confidence": 0.2}, {"confidence": 0.9}]
    state["embeddings"] = [np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)]
    state["properties"] = [
        {"stability": 0.0, "activity": 0.0, "uncertainty": 0.0},
        {"stability": 0.0, "activity": 0.0, "uncertainty": 0.0},
    ]

    updated = EvaluationAgent().run(state)
    assert updated["sequences"][0] == "AAAT"



def test_evaluation_agent_pae_weight_prefers_lower_pae() -> None:
    state = create_initial_state("AAAA")
    state["config"] = {
        "w_stability": 0.0,
        "w_activity": 0.0,
        "w_uncertainty": 0.0,
        "w_structure": 0.0,
        "w_plddt": 0.0,
        "w_ptm": 0.0,
        "w_pae": 1.0,
    }
    state["sequences"] = ["AAAA", "AAAT"]
    state["structures"] = [
        {"confidence": 0.0, "plddt_mean": 0.0, "ptm": 0.0, "pae_mean": 20.0},
        {"confidence": 0.0, "plddt_mean": 0.0, "ptm": 0.0, "pae_mean": 5.0},
    ]
    state["embeddings"] = [np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)]
    state["properties"] = [
        {"stability": 0.0, "activity": 0.0, "uncertainty": 0.0},
        {"stability": 0.0, "activity": 0.0, "uncertainty": 0.0},
    ]

    updated = EvaluationAgent().run(state)
    assert updated["sequences"][0] == "AAAT"



def test_optimization_agent_generates_next_population() -> None:
    state = create_initial_state("MKTFFV")
    state["iteration"] = 2
    state["config"] = {"top_k": 2, "num_candidates": 6, "mutation_rate": 1}
    state["sequences"] = ["MKTFFV", "MKTFFI", "MKTFFL"]
    state["structures"] = [{}, {}, {}]
    state["properties"] = [{}, {}, {}]
    state["scores"] = [0.9, 0.8, 0.7]

    updated = OptimizationAgent(random_seed=3).run(state)

    assert len(updated["next_sequences"]) == 6
    assert updated["next_sequences"][0] == "MKTFFV"
    assert updated["next_sequences"][1] == "MKTFFI"



def test_optimization_agent_respects_constraints_when_enabled() -> None:
    state = create_initial_state("AAAA")
    state["iteration"] = 0
    state["config"] = {
        "top_k": 2,
        "num_candidates": 4,
        "mutation_rate": 1,
        "selection_strategy": "elitist",
        "constraint_enabled": True,
        "constraint_mode": "hard",
        "min_stability": 0.7,
    }
    state["sequences"] = ["BAD", "GOOD", "OK"]
    state["scores"] = [0.9, 0.8, 0.7]
    state["structures"] = [{"confidence": 0.9}, {"confidence": 0.9}, {"confidence": 0.9}]
    state["properties"] = [
        {"stability": 0.2, "activity": 1.0},
        {"stability": 0.9, "activity": 1.0},
        {"stability": 0.8, "activity": 1.0},
    ]

    updated = OptimizationAgent(random_seed=1).run(state)
    assert updated["next_sequences"][0] == "GOOD"
    assert updated["next_sequences"][1] == "OK"
    assert updated["constraint_summary"]["passed"] == 2



def test_optimization_agent_soft_constraints_apply_penalty() -> None:
    state = create_initial_state("AAAA")
    state["iteration"] = 0
    state["config"] = {
        "top_k": 2,
        "num_candidates": 4,
        "mutation_rate": 1,
        "selection_strategy": "elitist",
        "constraint_enabled": True,
        "constraint_mode": "soft",
        "constraint_penalty": 1.0,
        "min_stability": 0.8,
    }
    state["sequences"] = ["SEQ1", "SEQ2", "SEQ3"]
    state["scores"] = [0.9, 0.8, 0.7]
    state["structures"] = [{}, {}, {}]
    state["properties"] = [
        {"stability": 0.1, "activity": 1.0},
        {"stability": 0.8, "activity": 1.0},
        {"stability": 0.75, "activity": 1.0},
    ]

    updated = OptimizationAgent(random_seed=1).run(state)
    assert updated["next_sequences"][0] == "SEQ2"
    assert updated["constraint_summary"]["mode"] == "soft"



def test_optimization_agent_diverse_strategy_respects_distance() -> None:
    state = create_initial_state("AAAA")
    state["iteration"] = 0
    state["config"] = {
        "top_k": 2,
        "num_candidates": 4,
        "mutation_rate": 1,
        "selection_strategy": "diverse",
        "min_hamming_distance": 2,
    }
    state["sequences"] = ["AAAA", "AAAT", "AATT", "TTTT"]
    state["scores"] = [0.9, 0.8, 0.7, 0.6]
    state["structures"] = [{}, {}, {}, {}]
    state["properties"] = [{}, {}, {}, {}]

    updated = OptimizationAgent(random_seed=1).run(state)
    elite_a, elite_b = updated["next_sequences"][0], updated["next_sequences"][1]
    assert hamming_distance(elite_a, elite_b) >= 2


def test_optimization_agent_pareto_selects_non_dominated_extremes() -> None:
    state = create_initial_state("AAAA")
    state["iteration"] = 0
    state["config"] = {
        "top_k": 2,
        "num_candidates": 4,
        "mutation_rate": 1,
        "selection_strategy": "pareto",
    }
    state["sequences"] = ["SEQ_A", "SEQ_B", "SEQ_C"]
    state["scores"] = [0.95, 0.94, 0.93]
    state["embeddings"] = [np.ones((4,), dtype=np.float32) for _ in range(3)]
    state["structures"] = [
        {"confidence": 0.9},
        {"confidence": 0.9},
        {"confidence": 0.5},
    ]
    state["properties"] = [
        {"stability": 0.9, "activity": 0.1, "uncertainty": 0.1},
        {"stability": 0.1, "activity": 0.9, "uncertainty": 0.1},
        {"stability": 0.5, "activity": 0.5, "uncertainty": 0.6},
    ]

    updated = OptimizationAgent(random_seed=1).run(state)
    elites = updated["next_sequences"][:2]
    assert "SEQ_A" in elites
    assert "SEQ_B" in elites


def test_optimization_agent_pareto_bo_generates_candidates() -> None:
    state = create_initial_state("MKTFFV")
    state["iteration"] = 0
    state["config"] = {
        "top_k": 2,
        "num_candidates": 6,
        "mutation_rate": 1,
        "selection_strategy": "pareto_bo",
        "bo_beta": 0.3,
        "bo_trials_per_parent": 6,
    }
    state["sequences"] = ["MKTFFV", "MKTFFI", "MKTFFL"]
    state["scores"] = [0.9, 0.85, 0.8]
    state["embeddings"] = [
        np.random.default_rng(1).normal(size=(8,)).astype(np.float32),
        np.random.default_rng(2).normal(size=(8,)).astype(np.float32),
        np.random.default_rng(3).normal(size=(8,)).astype(np.float32),
    ]
    state["structures"] = [{"confidence": 0.7}, {"confidence": 0.8}, {"confidence": 0.6}]
    state["properties"] = [
        {"stability": 0.9, "activity": 0.4, "uncertainty": 0.1},
        {"stability": 0.7, "activity": 0.8, "uncertainty": 0.2},
        {"stability": 0.6, "activity": 0.6, "uncertainty": 0.3},
    ]

    updated = OptimizationAgent(random_seed=11).run(state)
    assert len(updated["next_sequences"]) == 6
    assert len(set(updated["next_sequences"])) >= 3
