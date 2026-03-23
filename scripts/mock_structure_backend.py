"""Small mock external structure backend used for integration testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--backend", default="mock_external_backend")
    parser.add_argument("--confidence", type=float, default=0.87)
    args = parser.parse_args()

    sequence = Path(args.sequence_file).read_text(encoding="utf-8").strip()
    payload = {
        "engine": args.backend,
        "confidence": args.confidence,
        "sequence_length": len(sequence),
        "note": "mock external backend success",
    }

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
