"""Helpers for launching closed-loop campaign jobs from UI."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CampaignJobResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def build_campaign_command(
    config_path: str,
    data_path: str,
    output_dir: str,
    rounds: int,
    maple_iterations: int,
    acquisition_batch_size: int,
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
    selection_strategy: str = "pareto_bo",
    bo_beta: float = 0.3,
    bo_trials_per_parent: int = 8,
    num_candidates: int = 10,
    top_k: int = 3,
    mutation_rate: int = 1,
    seed: int = 42,
) -> list[str]:
    return [
        sys.executable,
        "scripts/closed_loop_campaign.py",
        "--config",
        config_path,
        "--data",
        data_path,
        "--output-dir",
        output_dir,
        "--rounds",
        str(rounds),
        "--maple-iterations",
        str(maple_iterations),
        "--acquisition-batch-size",
        str(acquisition_batch_size),
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
        "--selection-strategy",
        selection_strategy,
        "--bo-beta",
        str(bo_beta),
        "--bo-trials-per-parent",
        str(bo_trials_per_parent),
        "--num-candidates",
        str(num_candidates),
        "--top-k",
        str(top_k),
        "--mutation-rate",
        str(mutation_rate),
        "--seed",
        str(seed),
    ]


def run_campaign_job(
    root: Path,
    config_path: str,
    data_path: str,
    output_dir: str,
    rounds: int,
    maple_iterations: int,
    acquisition_batch_size: int,
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
    selection_strategy: str = "pareto_bo",
    bo_beta: float = 0.3,
    bo_trials_per_parent: int = 8,
    num_candidates: int = 10,
    top_k: int = 3,
    mutation_rate: int = 1,
    seed: int = 42,
) -> CampaignJobResult:
    cmd = build_campaign_command(
        config_path=config_path,
        data_path=data_path,
        output_dir=output_dir,
        rounds=rounds,
        maple_iterations=maple_iterations,
        acquisition_batch_size=acquisition_batch_size,
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
        selection_strategy=selection_strategy,
        bo_beta=bo_beta,
        bo_trials_per_parent=bo_trials_per_parent,
        num_candidates=num_candidates,
        top_k=top_k,
        mutation_rate=mutation_rate,
        seed=seed,
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
    return CampaignJobResult(
        command=cmd,
        returncode=int(proc.returncode),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
