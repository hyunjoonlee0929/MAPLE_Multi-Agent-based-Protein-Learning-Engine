from __future__ import annotations

from agents.evaluation_agent import EvaluationAgent
from agents.optimization_agent import OptimizationAgent
from agents.planner import PlannerAgent
from agents.property_agent import PropertyAgent
from agents.sequence_agent import SequenceAgent
from agents.structure_agent import StructureAgent
from core.pipeline import MaplePipeline, PipelineConfig
from core.state import create_initial_state



def test_pipeline_runs_multiple_iterations_end_to_end() -> None:
    state = create_initial_state("MKTFFVAVLGLCLLSQAS")
    state["config"] = {"num_candidates": 6, "top_k": 2, "mutation_rate": 1}

    pipeline = MaplePipeline(
        config=PipelineConfig(num_iterations=3),
        planner_agent=PlannerAgent(),
        sequence_agent=SequenceAgent(random_seed=1),
        structure_agent=StructureAgent(),
        property_agent=PropertyAgent(embedding_dim=16),
        optimization_agent=OptimizationAgent(random_seed=2),
        evaluation_agent=EvaluationAgent(),
    )

    final_state = pipeline.run(state)

    assert len(final_state["history"]) == 3
    assert len(final_state["sequences"]) == 6
    assert len(final_state["scores"]) == 6
    assert final_state["next_sequences"] is not None
    assert all(isinstance(score, float) for score in final_state["scores"])
