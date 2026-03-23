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
    val_ratio: float,
    split_seed: int,
    ridge_alphas: str,
    selection_strategy: str,
    bo_beta: float,
    bo_trials_per_parent: int,
    num_candidates: int,
    top_k: int,
    mutation_rate: int,
    seed: int,
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
        "--val-ratio",
        str(val_ratio),
        "--split-seed",
        str(split_seed),
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
    val_ratio: float,
    split_seed: int,
    ridge_alphas: str,
    selection_strategy: str,
    bo_beta: float,
    bo_trials_per_parent: int,
    num_candidates: int,
    top_k: int,
    mutation_rate: int,
    seed: int,
) -> CampaignJobResult:
    cmd = build_campaign_command(
        config_path=config_path,
        data_path=data_path,
        output_dir=output_dir,
        rounds=rounds,
        maple_iterations=maple_iterations,
        acquisition_batch_size=acquisition_batch_size,
        embedding_dim=embedding_dim,
        val_ratio=val_ratio,
        split_seed=split_seed,
        ridge_alphas=ridge_alphas,
        selection_strategy=selection_strategy,
        bo_beta=bo_beta,
        bo_trials_per_parent=bo_trials_per_parent,
        num_candidates=num_candidates,
        top_k=top_k,
        mutation_rate=mutation_rate,
        seed=seed,
    )
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
