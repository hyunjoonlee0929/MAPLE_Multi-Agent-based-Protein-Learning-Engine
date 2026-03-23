"""Evaluation agent for filtering, scoring, and ranking sequences."""

from __future__ import annotations

from utils.mutation import AMINO_ACIDS
from utils.scoring import combined_score_with_structure


class EvaluationAgent:
    """Computes normalized scores and maintains ranked state."""

    def _is_valid_sequence(self, sequence: str) -> bool:
        if not sequence:
            return False
        aa_set = set(AMINO_ACIDS)
        return all(char in aa_set for char in sequence)

    def run(self, state: dict) -> dict:
        sequences = state.get("sequences", [])
        structures = state.get("structures", [])
        embeddings = state.get("embeddings", [])
        properties = state.get("properties", [])

        aligned = list(zip(sequences, structures, embeddings, properties))
        aligned = [item for item in aligned if self._is_valid_sequence(item[0])]

        if not aligned:
            state["sequences"] = []
            state["structures"] = []
            state["embeddings"] = []
            state["properties"] = []
            state["scores"] = []
            state["history"].append(
                {
                    "iteration": int(state.get("iteration", 0)),
                    "best_sequence": None,
                    "best_score": None,
                    "mean_score": None,
                    "num_candidates": 0,
                }
            )
            return state

        config = state.get("config", {})
        w_stability = float(config.get("w_stability", 0.40))
        w_activity = float(config.get("w_activity", 0.40))
        w_uncertainty = float(config.get("w_uncertainty", 0.10))
        w_structure = float(config.get("w_structure", 0.10))

        stability = [item[3]["stability"] for item in aligned]
        activity = [item[3]["activity"] for item in aligned]
        uncertainty = [float(item[3].get("uncertainty", 0.0)) for item in aligned]
        structure_confidence = [float(item[1].get("confidence", 0.0)) for item in aligned]

        scores = combined_score_with_structure(
            stability=stability,
            activity=activity,
            uncertainty=uncertainty,
            structure_confidence=structure_confidence,
            w_stability=w_stability,
            w_activity=w_activity,
            w_uncertainty=w_uncertainty,
            w_structure=w_structure,
        )

        ranked = sorted(
            zip(aligned, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )

        state["sequences"] = [item[0][0] for item in ranked]
        state["structures"] = [item[0][1] for item in ranked]
        state["embeddings"] = [item[0][2] for item in ranked]
        state["properties"] = [item[0][3] for item in ranked]
        state["scores"] = [float(item[1]) for item in ranked]

        state["history"].append(
            {
                "iteration": int(state.get("iteration", 0)),
                "best_sequence": state["sequences"][0],
                "best_score": state["scores"][0],
                "best_uncertainty": float(state["properties"][0].get("uncertainty", 0.0)),
                "best_structure_confidence": float(state["structures"][0].get("confidence", 0.0)),
                "mean_score": float(sum(state["scores"]) / len(state["scores"])),
                "num_candidates": len(state["sequences"]),
            }
        )
        return state
