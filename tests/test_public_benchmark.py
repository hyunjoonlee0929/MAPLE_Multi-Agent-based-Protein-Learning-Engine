from __future__ import annotations

import json
from pathlib import Path

from core.public_benchmark import aggregate_rows, benchmark_markdown, load_benchmark_manifest


def test_load_benchmark_manifest_reads_datasets(tmp_path: Path) -> None:
    manifest = {
        "datasets": [
            {
                "name": "D1",
                "path": "data/d1.csv",
                "source_url": "https://example.org/d1",
                "license": "MIT",
            }
        ]
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    out = load_benchmark_manifest(p)
    assert len(out) == 1
    assert out[0]["name"] == "D1"
    assert out[0]["path"] == "data/d1.csv"


def test_aggregate_rows_sorts_by_rmse() -> None:
    rows = [
        {"dataset": "A", "val_rmse_mean": 0.4},
        {"dataset": "B", "val_rmse_mean": 0.2},
    ]
    out = aggregate_rows(rows)
    assert out[0]["dataset"] == "B"
    assert out[0]["rank"] == 1
    assert out[1]["rank"] == 2


def test_benchmark_markdown_has_table() -> None:
    rows = [
        {
            "rank": 1,
            "dataset": "D1",
            "embedding_backend": "random",
            "split_mode": "scaffold",
            "val_rmse_mean": 0.1,
            "val_rmse_std": 0.01,
            "val_mae_mean": 0.1,
            "val_mae_std": 0.01,
            "val_r2_mean": 0.8,
            "val_r2_std": 0.02,
            "val_ece_mean": 0.05,
            "val_ece_std": 0.01,
            "best_alpha_mode": 0.001,
        }
    ]
    text = benchmark_markdown(rows)
    assert "| Dataset | Embedding |" in text
    assert "D1" in text
