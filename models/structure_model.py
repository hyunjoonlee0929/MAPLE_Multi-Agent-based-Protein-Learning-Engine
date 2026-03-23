"""Structure prediction backends for MAPLE."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol


class StructurePredictorLike(Protocol):
    """Interface for structure prediction backends."""

    def predict(self, sequence: str) -> dict:
        """Return a serializable structure representation."""



def _pseudo_confidence(sequence: str, salt: str) -> float:
    digest = hashlib.sha1(f"{salt}:{sequence}".encode("utf-8")).hexdigest()
    return round(int(digest[:2], 16) / 255.0, 4)



def _mock_structure(sequence: str, backend: str, mode: str) -> dict:
    return {
        "sequence_length": len(sequence),
        "backend": backend,
        "mode": mode,
        "confidence": _pseudo_confidence(sequence, backend),
        "note": "adapter mock output; integrate external engine for real folding",
    }


class DummyStructurePredictor:
    """MVP backend with deterministic pseudo-confidence."""

    def predict(self, sequence: str) -> dict:
        return _mock_structure(sequence, backend="dummy_structure_predictor", mode="mock")


class _ExternalToolAdapter:
    """Optional external command adapter for structure prediction backends."""

    def __init__(self, command: str | None = None) -> None:
        self.command = command.strip() if command else ""

    def run_external(self, sequence: str) -> dict | None:
        if not self.command:
            return None

        cmd_head = self.command.split()[0]
        if shutil.which(cmd_head) is None:
            return None

        with tempfile.TemporaryDirectory(prefix="maple_struct_") as tmp:
            seq_file = Path(tmp) / "sequence.txt"
            out_file = Path(tmp) / "result.json"
            seq_file.write_text(sequence, encoding="utf-8")

            expanded = self.command.format(sequence_file=str(seq_file), output_file=str(out_file))
            proc = subprocess.run(expanded, shell=True, capture_output=True, text=True)
            if proc.returncode != 0 or not out_file.exists():
                return None

            output_text = out_file.read_text(encoding="utf-8").strip()
            return {
                "sequence_length": len(sequence),
                "backend": "external_structure_tool",
                "mode": "external",
                "confidence": _pseudo_confidence(sequence, "external"),
                "command": expanded,
                "external_result": output_text[:1000],
            }


class ESMFoldStructurePredictor(_ExternalToolAdapter):
    """ESMFold adapter with optional external command and robust mock fallback."""

    def __init__(self, command: str | None = None) -> None:
        super().__init__(command=command)

    def predict(self, sequence: str) -> dict:
        external = self.run_external(sequence)
        if external is not None:
            external["backend"] = "esmfold_adapter"
            return external
        return _mock_structure(sequence, backend="esmfold_adapter", mode="mock")


class AlphaFoldStructurePredictor(_ExternalToolAdapter):
    """AlphaFold2 adapter with optional external command and robust mock fallback."""

    def __init__(self, command: str | None = None) -> None:
        super().__init__(command=command)

    def predict(self, sequence: str) -> dict:
        external = self.run_external(sequence)
        if external is not None:
            external["backend"] = "alphafold2_adapter"
            return external
        return _mock_structure(sequence, backend="alphafold2_adapter", mode="mock")



def build_structure_predictor(
    backend: str = "dummy",
    options: dict | None = None,
) -> StructurePredictorLike:
    options = options or {}
    normalized = backend.strip().lower()

    if normalized == "dummy":
        return DummyStructurePredictor()
    if normalized == "esmfold":
        return ESMFoldStructurePredictor(command=options.get("esmfold_command"))
    if normalized == "alphafold2":
        return AlphaFoldStructurePredictor(command=options.get("alphafold2_command"))

    raise ValueError(f"Unsupported structure backend: {backend}")
