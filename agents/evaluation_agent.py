"""Evaluation agent for filtering, scoring, and ranking sequences."""

from __future__ import annotations

from utils.mutation import AMINO_ACIDS
from utils.scoring import combined_score_with_structure_quality


class EvaluationAgent:
    """Computes normalized scores and maintains ranked state."""

    def _is_valid_sequence(self, sequence: str) -> bool:
        if not sequence:
            return False
        aa_set = set(AMINO_ACIDS)
        return all(char in aa_set for char in sequence)

    def _structure_signals(self, structure: dict) -> tuple[float, float, float, float]:
        confidence = float(structure.get("confidence", 0.0))

        # Fallbacks keep pipeline robust even when some structure metrics are missing.
        plddt_mean = float(structure.get("plddt_mean", confidence * 100.0))
        ptm = float(structure.get("ptm", confidence))
        pae_mean = float(structure.get("pae_mean", max(0.0, (1.0 - confidence) * 30.0)))

        return confidence, plddt_mean, ptm, pae_mean

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
        w_stability = float(config.get("w_stability", 0.35))
        w_activity = float(config.get("w_activity", 0.35))
        w_uncertainty = float(config.get("w_uncertainty", 0.10))
        w_structure = float(config.get("w_structure", 0.10))
        w_plddt = float(config.get("w_plddt", 0.05))
        w_ptm = float(config.get("w_ptm", 0.03))
        w_pae = float(config.get("w_pae", 0.02))

        stability = [item[3]["stability"] for item in aligned]
        activity = [item[3]["activity"] for item in aligned]
        uncertainty = [float(item[3].get("uncertainty", 0.0)) for item in aligned]

        structure_confidence: list[float] = []
        plddt_mean: list[float] = []
        ptm: list[float] = []
        pae_mean: list[float] = []
        for item in aligned:
            c, plddt, ptm_val, pae = self._structure_signals(item[1])
            structure_confidence.append(c)
            plddt_mean.append(plddt)
            ptm.append(ptm_val)
            pae_mean.append(pae)

        scores = combined_score_with_structure_quality(
            stability=stability,
            activity=activity,
            uncertainty=uncertainty,
            structure_confidence=structure_confidence,
            plddt_mean=plddt_mean,
            ptm=ptm,
            pae_mean=pae_mean,
            w_stability=w_stability,
            w_activity=w_activity,
            w_uncertainty=w_uncertainty,
            w_structure=w_structure,
            w_plddt=w_plddt,
            w_ptm=w_ptm,
            w_pae=w_pae,
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

        best_structure = state["structures"][0] if state["structures"] else {}
        state["history"].append(
            {
                "iteration": int(state.get("iteration", 0)),
                "best_sequence": state["sequences"][0],
                "best_score": state["scores"][0],
                "best_uncertainty": float(state["properties"][0].get("uncertainty", 0.0)),
                "best_structure_confidence": float(best_structure.get("confidence", 0.0)),
                "best_plddt_mean": float(best_structure.get("plddt_mean", 0.0)),
                "best_ptm": float(best_structure.get("ptm", 0.0)),
                "best_pae_mean": float(best_structure.get("pae_mean", 0.0)),
                "mean_score": float(sum(state["scores"]) / len(state["scores"])),
                "num_candidates": len(state["sequences"]),
            }
        )
        return state
