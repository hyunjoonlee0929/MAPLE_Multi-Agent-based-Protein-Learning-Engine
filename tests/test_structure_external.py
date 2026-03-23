from __future__ import annotations

from pathlib import Path

from models.structure_model import build_structure_predictor



def test_esmfold_external_command_mode() -> None:
    root = Path("/Users/hyunjoon/codex/MAPLE")
    command = (
        f"python3 '{root / 'scripts' / 'mock_structure_backend.py'}' "
        "--sequence-file {sequence_file} --output-file {output_file} --backend test_esm --confidence 0.91"
    )

    predictor = build_structure_predictor(
        "esmfold",
        options={
            "esmfold_command": command,
            "structure_timeout_sec": 30,
            "structure_retries": 0,
        },
    )
    out = predictor.predict("MKTFFV")

    assert out["backend"] == "esmfold_adapter"
    assert out["mode"] == "external"
    assert abs(float(out["confidence"]) - 0.91) < 1e-8
    assert out["engine"] == "test_esm"
    assert "plddt_mean" in out
    assert "ptm" in out
    assert "pae_mean" in out
    assert "pdb_path" in out



def test_alphafold_external_command_mode() -> None:
    root = Path("/Users/hyunjoon/codex/MAPLE")
    command = (
        f"python3 '{root / 'scripts' / 'mock_structure_backend.py'}' "
        "--sequence-file {sequence_file} --output-file {output_file} --backend test_af2 --confidence 0.77"
    )

    predictor = build_structure_predictor(
        "alphafold2",
        options={
            "alphafold2_command": command,
            "structure_timeout_sec": 30,
            "structure_retries": 0,
        },
    )
    out = predictor.predict("MKTFFV")

    assert out["backend"] == "alphafold2_adapter"
    assert out["mode"] == "external"
    assert abs(float(out["confidence"]) - 0.77) < 1e-8
    assert out["engine"] == "test_af2"


def test_esmfold_strict_raises_when_command_missing() -> None:
    predictor = build_structure_predictor(
        "esmfold",
        options={
            "esmfold_command": "",
            "structure_strict": True,
        },
    )
    try:
        predictor.predict("MKTFFV")
        raised = False
    except RuntimeError as exc:
        raised = True
        assert "not configured" in str(exc)
    assert raised is True
