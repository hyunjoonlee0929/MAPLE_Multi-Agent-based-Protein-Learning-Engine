"""Helpers for launching DBTL ingestion + retraining jobs from UI."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DbtlJobResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def build_dbtl_ingest_command(
    seed_data: str,
    dbtl_input: str,
    dbtl_format: str,
    output_dir: str,
    checkpoint_out: str,
    embedding_dim: int,
    embedding_backend: str = "random",
    embedding_model_id: str = "",
    embedding_device: str = "cpu",
    embedding_pooling: str = "mean",
    embedding_allow_mock: bool = True,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    split_mode: str = "random",
    scaffold_k: int = 3,
    ensemble_size: int = 1,
    ece_bins: int = 10,
    ridge_alphas: str = "1e-4,1e-3,1e-2,1e-1",
    min_imported_records: int = 1,
) -> list[str]:
    return [
        sys.executable,
        "scripts/dbtl_ingest_retrain.py",
        "--seed-data",
        seed_data,
        "--dbtl-input",
        dbtl_input,
        "--dbtl-format",
        dbtl_format,
        "--output-dir",
        output_dir,
        "--checkpoint-out",
        checkpoint_out,
        "--embedding-dim",
        str(embedding_dim),
        "--embedding-backend",
        embedding_backend,
        "--embedding-model-id",
        embedding_model_id,
        "--embedding-device",
        embedding_device,
        "--embedding-pooling",
        embedding_pooling,
        "--val-ratio",
        str(val_ratio),
        "--split-seed",
        str(split_seed),
        "--split-mode",
        split_mode,
        "--scaffold-k",
        str(scaffold_k),
        "--ensemble-size",
        str(ensemble_size),
        "--ece-bins",
        str(ece_bins),
        "--ridge-alphas",
        ridge_alphas,
        "--min-imported-records",
        str(min_imported_records),
    ]


def run_dbtl_ingest_job(
    root: Path,
    seed_data: str,
    dbtl_input: str,
    dbtl_format: str,
    output_dir: str,
    checkpoint_out: str,
    embedding_dim: int,
    embedding_backend: str = "random",
    embedding_model_id: str = "",
    embedding_device: str = "cpu",
    embedding_pooling: str = "mean",
    embedding_allow_mock: bool = True,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    split_mode: str = "random",
    scaffold_k: int = 3,
    ensemble_size: int = 1,
    ece_bins: int = 10,
    ridge_alphas: str = "1e-4,1e-3,1e-2,1e-1",
    min_imported_records: int = 1,
) -> DbtlJobResult:
    cmd = build_dbtl_ingest_command(
        seed_data=seed_data,
        dbtl_input=dbtl_input,
        dbtl_format=dbtl_format,
        output_dir=output_dir,
        checkpoint_out=checkpoint_out,
        embedding_dim=embedding_dim,
        embedding_backend=embedding_backend,
        embedding_model_id=embedding_model_id,
        embedding_device=embedding_device,
        embedding_pooling=embedding_pooling,
        embedding_allow_mock=embedding_allow_mock,
        val_ratio=val_ratio,
        split_seed=split_seed,
        split_mode=split_mode,
        scaffold_k=scaffold_k,
        ensemble_size=ensemble_size,
        ece_bins=ece_bins,
        ridge_alphas=ridge_alphas,
        min_imported_records=min_imported_records,
    )
    if not embedding_allow_mock:
        cmd.append("--disable-embedding-mock-fallback")
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    return DbtlJobResult(
        command=cmd,
        returncode=int(proc.returncode),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
