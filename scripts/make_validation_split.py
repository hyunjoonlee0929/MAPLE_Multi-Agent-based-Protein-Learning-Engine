"""Create a reusable fixed validation split index file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_property_numpy import load_dataset, split_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed validation split indices")
    parser.add_argument("--data", type=str, default="data/sample_property_labels.csv")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-mode", type=str, default="random", help="Split strategy: random|scaffold")
    parser.add_argument("--scaffold-k", type=int, default=3, help="Scaffold key size for split_mode=scaffold")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/property_validation/fixed_val_split.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    sequences, _ = load_dataset(data_path)
    train_idx, val_idx = split_indices(
        n=len(sequences),
        val_ratio=float(args.val_ratio),
        seed=int(args.split_seed),
        split_mode=str(args.split_mode),
        sequences=sequences,
        scaffold_k=int(args.scaffold_k),
    )

    payload = {
        "dataset": str(data_path),
        "num_samples": len(sequences),
        "val_ratio": float(args.val_ratio),
        "split_seed": int(args.split_seed),
        "split_mode": str(args.split_mode),
        "scaffold_k": int(args.scaffold_k),
        "train_indices": train_idx.astype(int).tolist(),
        "val_indices": val_idx.astype(int).tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved fixed validation split: {output_path}")
    print(f"train_count={len(payload['train_indices'])}, val_count={len(payload['val_indices'])}")


if __name__ == "__main__":
    main()
