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
    state["config"] = {"w_stability": 0.45, "w_activity": 0.45, "w_uncertainty": 0.10}
    state["sequences"] = ["MKTFFV", "INVALIDX"]
    state["structures"] = [{"id": 1}, {"id": 2}]
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
    state["config"] = {"w_stability": 0.0, "w_activity": 0.0, "w_uncertainty": 1.0}
    state["sequences"] = ["AAAA", "AAAT"]
    state["structures"] = [{}, {}]
    state["embeddings"] = [np.zeros((2,), dtype=np.float32), np.zeros((2,), dtype=np.float32)]
    state["properties"] = [
        {"stability": 0.0, "activity": 0.0, "uncertainty": 0.1},
        {"stability": 0.0, "activity": 0.0, "uncertainty": 0.9},
    ]

    updated = EvaluationAgent().run(state)
    assert updated["sequences"][0] == "AAAT"



def test_optimization_agent_generates_next_population() -> None:
    state = create_initial_state("MKTFFV")
    state["iteration"] = 2
    state["config"] = {"top_k": 2, "num_candidates": 6, "mutation_rate": 1}
    state["sequences"] = ["MKTFFV", "MKTFFI", "MKTFFL"]

    updated = OptimizationAgent(random_seed=3).run(state)

    assert len(updated["next_sequences"]) == 6
    assert updated["next_sequences"][0] == "MKTFFV"
    assert updated["next_sequences"][1] == "MKTFFI"



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

    updated = OptimizationAgent(random_seed=1).run(state)
    elite_a, elite_b = updated["next_sequences"][0], updated["next_sequences"][1]
    assert hamming_distance(elite_a, elite_b) >= 2
