from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.train_property_numpy import (
    load_dataset,
    split_indices,
    split_train_val,
    split_train_val_with_indices,
)



def test_load_dataset_reads_sequences_and_targets() -> None:
    csv_path = Path("/Users/hyunjoon/codex/MAPLE/data/sample_property_labels.csv")
    sequences, targets = load_dataset(csv_path)
    assert len(sequences) > 0
    assert targets.shape[1] == 2


def test_split_train_val_produces_non_empty_splits() -> None:
    sequences = ["AAAA", "AAAT", "AATA", "ATAA", "TAAA"]
    targets = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
            [0.4, 0.5],
            [0.5, 0.6],
        ],
        dtype=np.float32,
    )
    train_seq, train_t, val_seq, val_t = split_train_val(sequences, targets, val_ratio=0.4, seed=7)
    assert len(train_seq) > 0
    assert len(val_seq) > 0
    assert train_t.shape[1] == 2
    assert val_t.shape[1] == 2
    assert len(train_seq) + len(val_seq) == len(sequences)


def test_split_indices_are_deterministic_and_disjoint() -> None:
    train_idx_1, val_idx_1 = split_indices(n=20, val_ratio=0.2, seed=42)
    train_idx_2, val_idx_2 = split_indices(n=20, val_ratio=0.2, seed=42)
    assert np.array_equal(train_idx_1, train_idx_2)
    assert np.array_equal(val_idx_1, val_idx_2)
    assert set(train_idx_1.tolist()).isdisjoint(set(val_idx_1.tolist()))


def test_split_train_val_with_indices_respects_given_indices() -> None:
    sequences = ["A", "B", "C", "D"]
    targets = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.float32)
    train_idx = np.array([2, 0], dtype=np.int64)
    val_idx = np.array([3, 1], dtype=np.int64)
    train_seq, train_t, val_seq, val_t = split_train_val_with_indices(sequences, targets, train_idx, val_idx)
    assert train_seq == ["C", "A"]
    assert val_seq == ["D", "B"]
    assert train_t.tolist() == [[3.0, 3.0], [1.0, 1.0]]
    assert val_t.tolist() == [[4.0, 4.0], [2.0, 2.0]]


def test_scaffold_split_is_deterministic_and_disjoint() -> None:
    sequences = ["AAAAAA", "AAAATA", "VVVVVV", "VVVVAV", "DDDDDD", "DDDDED"]
    train_idx_1, val_idx_1 = split_indices(
        n=len(sequences),
        val_ratio=0.34,
        seed=11,
        split_mode="scaffold",
        sequences=sequences,
        scaffold_k=2,
    )
    train_idx_2, val_idx_2 = split_indices(
        n=len(sequences),
        val_ratio=0.34,
        seed=11,
        split_mode="scaffold",
        sequences=sequences,
        scaffold_k=2,
    )
    assert np.array_equal(train_idx_1, train_idx_2)
    assert np.array_equal(val_idx_1, val_idx_2)
    assert set(train_idx_1.tolist()).isdisjoint(set(val_idx_1.tolist()))
