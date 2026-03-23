from __future__ import annotations

from core.campaign_jobs import build_campaign_command


def test_build_campaign_command_contains_expected_args() -> None:
    cmd = build_campaign_command(
        config_path="config.yaml",
        data_path="data/sample_property_labels.csv",
        output_dir="outputs/closed_loop_campaign",
        rounds=3,
        maple_iterations=3,
        acquisition_batch_size=4,
        embedding_dim=128,
        val_ratio=0.2,
        split_seed=42,
        ridge_alphas="1e-4,1e-3",
        selection_strategy="pareto_bo",
        bo_beta=0.3,
        bo_trials_per_parent=8,
        num_candidates=10,
        top_k=3,
        mutation_rate=1,
        seed=42,
    )
    assert "scripts/closed_loop_campaign.py" in cmd
    assert "--rounds" in cmd
    assert "3" in cmd
