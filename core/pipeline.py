"""Core MAPLE multi-agent execution pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agents.evaluation_agent import EvaluationAgent
from agents.optimization_agent import OptimizationAgent
from agents.planner import PlannerAgent
from agents.property_agent import PropertyAgent
from agents.sequence_agent import SequenceAgent
from agents.structure_agent import StructureAgent
from core.state import State, ensure_numpy_embeddings


@dataclass
class PipelineConfig:
    num_iterations: int = 5


class MaplePipeline:
    """Runs the MAPLE multi-agent optimization loop."""

    def __init__(
        self,
        config: PipelineConfig,
        planner_agent: PlannerAgent,
        sequence_agent: SequenceAgent,
        structure_agent: StructureAgent,
        property_agent: PropertyAgent,
        optimization_agent: OptimizationAgent,
        evaluation_agent: EvaluationAgent,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.planner_agent = planner_agent
        self.sequence_agent = sequence_agent
        self.structure_agent = structure_agent
        self.property_agent = property_agent
        self.optimization_agent = optimization_agent
        self.evaluation_agent = evaluation_agent
        self.logger = logger or logging.getLogger(__name__)

    def run(self, state: State) -> State:
        for iteration in range(self.config.num_iterations):
            state["iteration"] = iteration

            state = self.planner_agent.run(state)
            state = self.sequence_agent.run(state)
            state = self.structure_agent.run(state)
            state = self.property_agent.run(state)
            state = self.evaluation_agent.run(state)
            ensure_numpy_embeddings(state)
            state = self.optimization_agent.run(state)

            best_seq = state["history"][-1]["best_sequence"]
            best_score = state["history"][-1]["best_score"]
            self.logger.info(
                "Iteration %d complete | best_score=%.4f | best_sequence=%s",
                iteration,
                0.0 if best_score is None else best_score,
                best_seq,
            )

        return state
