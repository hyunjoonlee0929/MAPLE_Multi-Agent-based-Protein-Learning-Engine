"""Structure prediction backends for MAPLE."""

from __future__ import annotations

import hashlib
from typing import Protocol


class StructurePredictorLike(Protocol):
    """Interface for structure prediction backends."""

    def predict(self, sequence: str) -> dict:
        """Return a serializable structure representation."""


class DummyStructurePredictor:
    """MVP backend with deterministic pseudo-confidence."""

    def predict(self, sequence: str) -> dict:
        digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()
        pseudo_confidence = int(digest[:2], 16) / 255.0
        return {
            "sequence_length": len(sequence),
            "backend": "dummy_structure_predictor",
            "confidence": round(pseudo_confidence, 4),
        }


class ESMFoldStructurePredictor:
    """Adapter placeholder for future ESMFold integration."""

    def predict(self, sequence: str) -> dict:
        raise NotImplementedError("ESMFold integration is not implemented in MVP")


class AlphaFoldStructurePredictor:
    """Adapter placeholder for future AlphaFold2 integration."""

    def predict(self, sequence: str) -> dict:
        raise NotImplementedError("AlphaFold2 integration is not implemented in MVP")



def build_structure_predictor(backend: str = "dummy") -> StructurePredictorLike:
    normalized = backend.strip().lower()
    if normalized == "dummy":
        return DummyStructurePredictor()
    if normalized == "esmfold":
        return ESMFoldStructurePredictor()
    if normalized == "alphafold2":
        return AlphaFoldStructurePredictor()
    raise ValueError(f"Unsupported structure backend: {backend}")
