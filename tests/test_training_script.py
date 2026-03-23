from __future__ import annotations

from pathlib import Path

from scripts.train_property_numpy import load_dataset



def test_load_dataset_reads_sequences_and_targets() -> None:
    csv_path = Path("/Users/hyunjoon/codex/MAPLE/data/sample_property_labels.csv")
    sequences, targets = load_dataset(csv_path)
    assert len(sequences) > 0
    assert targets.shape[1] == 2
