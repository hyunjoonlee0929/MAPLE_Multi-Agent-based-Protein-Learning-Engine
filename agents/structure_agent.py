"""Structure agent with a placeholder structure prediction backend."""

from __future__ import annotations

import hashlib


class StructureAgent:
    """Produces dummy structure outputs compatible with future real predictors."""

    def run(self, state: dict) -> dict:
        structures = []
        for sequence in state.get("sequences", []):
            digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()
            pseudo_confidence = int(digest[:2], 16) / 255.0
            structures.append(
                {
                    "sequence_length": len(sequence),
                    "backend": "dummy_structure_predictor",
                    "confidence": round(pseudo_confidence, 4),
                }
            )

        state["structures"] = structures
        return state
