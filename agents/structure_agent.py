"""Structure agent with configurable structure prediction backend."""

from __future__ import annotations

from models.structure_model import StructurePredictorLike, build_structure_predictor


class StructureAgent:
    """Produces structure outputs compatible with future real predictors."""

    def __init__(
        self,
        backend: str = "dummy",
        options: dict | None = None,
        batch_size: int = 16,
    ) -> None:
        self.backend = backend
        self.options = options or {}
        self.batch_size = max(1, int(batch_size))
        self.predictor: StructurePredictorLike = build_structure_predictor(backend, options=self.options)

    def _fallback_structure(self, sequence: str, error: str) -> dict:
        return {
            "sequence_length": len(sequence),
            "backend": f"{self.backend}_adapter",
            "mode": "error_fallback",
            "confidence": 0.0,
            "engine": "fallback",
            "note": error[:500],
        }

    def run(self, state: dict) -> dict:
        sequences = state.get("sequences", [])
        outputs: list[dict] = []

        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]
            for seq in batch:
                try:
                    outputs.append(self.predictor.predict(seq))
                except Exception as exc:
                    outputs.append(self._fallback_structure(seq, str(exc)))

        state["structures"] = outputs
        return state
