"""Evaluation agent for filtering, scoring, and ranking sequences."""

from __future__ import annotations

from utils.mutation import AMINO_ACIDS
from utils.scoring import combined_score


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

        stability = [item[3]["stability"] for item in aligned]
        activity = [item[3]["activity"] for item in aligned]
        scores = combined_score(stability, activity)

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
                "mean_score": float(sum(state["scores"]) / len(state["scores"])),
                "num_candidates": len(state["sequences"]),
            }
        )
        return state
