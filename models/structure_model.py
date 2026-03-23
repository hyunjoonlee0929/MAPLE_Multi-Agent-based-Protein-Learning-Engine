"""Structure prediction backends for MAPLE."""

from __future__ import annotations

import hashlib
import json
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
        "engine": "mock",
        "note": "adapter mock output; integrate external engine for real folding",
    }



def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class DummyStructurePredictor:
    """MVP backend with deterministic pseudo-confidence."""

    def predict(self, sequence: str) -> dict:
        return _mock_structure(sequence, backend="dummy_structure_predictor", mode="mock")


class _ExternalToolAdapter:
    """Optional external command adapter for structure prediction backends."""

    def __init__(
        self,
        adapter_backend: str,
        command: str | None = None,
        timeout_sec: int = 60,
        retries: int = 0,
    ) -> None:
        self.adapter_backend = adapter_backend
        self.command = command.strip() if command else ""
        self.timeout_sec = max(1, int(timeout_sec))
        self.retries = max(0, int(retries))

    def _normalize_external_payload(self, sequence: str, expanded_command: str, payload: dict) -> dict:
        confidence = _safe_float(payload.get("confidence"))
        if confidence is None:
            confidence = _pseudo_confidence(sequence, f"{self.adapter_backend}:external")

        normalized = {
            "sequence_length": len(sequence),
            "backend": self.adapter_backend,
            "mode": "external",
            "confidence": float(confidence),
            "engine": str(payload.get("engine", "external_command")),
            "command": expanded_command,
        }

        for key in [
            "model_id",
            "pdb_path",
            "pae_mean",
            "ptm",
            "plddt_mean",
            "runtime_sec",
            "note",
        ]:
            if key in payload:
                normalized[key] = payload[key]

        return normalized

    def _read_payload(self, out_file: Path) -> dict | None:
        text = out_file.read_text(encoding="utf-8").strip()
        if not text:
            return None

        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
            return {"raw_output": payload}
        except json.JSONDecodeError:
            return {"raw_output": text[:1000]}

    def run_external(self, sequence: str) -> dict | None:
        if not self.command:
            return None

        cmd_head = self.command.split()[0]
        if shutil.which(cmd_head) is None:
            return None

        for _attempt in range(self.retries + 1):
            with tempfile.TemporaryDirectory(prefix="maple_struct_") as tmp:
                seq_file = Path(tmp) / "sequence.txt"
                out_file = Path(tmp) / "result.json"
                seq_file.write_text(sequence, encoding="utf-8")

                expanded = self.command.format(
                    sequence_file=str(seq_file),
                    output_file=str(out_file),
                )

                try:
                    proc = subprocess.run(
                        expanded,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_sec,
                    )
                except subprocess.TimeoutExpired:
                    continue

                if proc.returncode != 0 or not out_file.exists():
                    continue

                payload = self._read_payload(out_file)
                if payload is None:
                    continue

                return self._normalize_external_payload(sequence, expanded, payload)

        return None


class ESMFoldStructurePredictor(_ExternalToolAdapter):
    """ESMFold adapter with optional external command and robust mock fallback."""

    def __init__(
        self,
        command: str | None = None,
        timeout_sec: int = 60,
        retries: int = 0,
    ) -> None:
        super().__init__(
            adapter_backend="esmfold_adapter",
            command=command,
            timeout_sec=timeout_sec,
            retries=retries,
        )

    def predict(self, sequence: str) -> dict:
        external = self.run_external(sequence)
        if external is not None:
            return external
        return _mock_structure(sequence, backend="esmfold_adapter", mode="mock")


class AlphaFoldStructurePredictor(_ExternalToolAdapter):
    """AlphaFold2 adapter with optional external command and robust mock fallback."""

    def __init__(
        self,
        command: str | None = None,
        timeout_sec: int = 60,
        retries: int = 0,
    ) -> None:
        super().__init__(
            adapter_backend="alphafold2_adapter",
            command=command,
            timeout_sec=timeout_sec,
            retries=retries,
        )

    def predict(self, sequence: str) -> dict:
        external = self.run_external(sequence)
        if external is not None:
            return external
        return _mock_structure(sequence, backend="alphafold2_adapter", mode="mock")



def build_structure_predictor(
    backend: str = "dummy",
    options: dict | None = None,
) -> StructurePredictorLike:
    options = options or {}
    normalized = backend.strip().lower()

    timeout_sec = int(options.get("structure_timeout_sec", 60) or 60)
    retries = int(options.get("structure_retries", 0) or 0)

    if normalized == "dummy":
        return DummyStructurePredictor()
    if normalized == "esmfold":
        return ESMFoldStructurePredictor(
            command=options.get("esmfold_command"),
            timeout_sec=timeout_sec,
            retries=retries,
        )
    if normalized == "alphafold2":
        return AlphaFoldStructurePredictor(
            command=options.get("alphafold2_command"),
            timeout_sec=timeout_sec,
            retries=retries,
        )

    raise ValueError(f"Unsupported structure backend: {backend}")
