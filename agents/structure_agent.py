"""Structure agent with configurable structure prediction backend."""

from __future__ import annotations

from models.structure_model import StructurePredictorLike, build_structure_predictor


class StructureAgent:
    """Produces structure outputs compatible with future real predictors."""

    def __init__(self, backend: str = "dummy", options: dict | None = None) -> None:
        self.predictor: StructurePredictorLike = build_structure_predictor(backend, options=options)

    def run(self, state: dict) -> dict:
        state["structures"] = [self.predictor.predict(seq) for seq in state.get("sequences", [])]
        return state
