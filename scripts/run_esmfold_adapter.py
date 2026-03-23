"""External ESMFold adapter command for MAPLE structure backend."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path



def _mock_payload(sequence: str, model_id: str, note: str) -> dict:
    return {
        "engine": "esmfold_mock_runner",
        "model_id": model_id,
        "confidence": 0.5,
        "plddt_mean": 50.0,
        "ptm": None,
        "pae_mean": None,
        "pdb_path": None,
        "runtime_sec": 0.0,
        "note": note,
        "sequence_length": len(sequence),
    }



def run_real_esmfold(sequence: str, model_id: str) -> dict:
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = EsmForProteinFolding.from_pretrained(model_id)
    model.eval()

    tokens = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        outputs = model(**tokens)

    plddt_mean = None
    ptm = None
    pae_mean = None
    if hasattr(outputs, "plddt"):
        plddt = outputs.plddt
        plddt_mean = float(plddt.mean().detach().cpu().item())
    if hasattr(outputs, "ptm") and outputs.ptm is not None:
        ptm = float(outputs.ptm.mean().detach().cpu().item())
    if hasattr(outputs, "predicted_aligned_error") and outputs.predicted_aligned_error is not None:
        pae_mean = float(outputs.predicted_aligned_error.mean().detach().cpu().item())

    runtime_sec = round(time.time() - started, 4)
    confidence = (plddt_mean / 100.0) if plddt_mean is not None else 0.5

    payload = {
        "engine": "transformers_esmfold",
        "model_id": model_id,
        "confidence": float(confidence),
        "plddt_mean": plddt_mean,
        "ptm": ptm,
        "pae_mean": pae_mean,
        "pdb_path": None,
        "runtime_sec": runtime_sec,
        "sequence_length": len(sequence),
    }
    return payload



def main() -> None:
    parser = argparse.ArgumentParser(description="Run ESMFold adapter and emit normalized JSON")
    parser.add_argument("--sequence-file", required=True, type=str, help="Input sequence file path")
    parser.add_argument("--output-file", required=True, type=str, help="Output JSON file path")
    parser.add_argument("--model-id", type=str, default="facebook/esmfold_v1", help="HF model id")
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="If real ESMFold runtime is unavailable, emit mock JSON instead of failing",
    )
    args = parser.parse_args()

    sequence = Path(args.sequence_file).read_text(encoding="utf-8").strip()

    try:
        payload = run_real_esmfold(sequence=sequence, model_id=args.model_id)
    except Exception as exc:
        if not args.allow_mock:
            raise
        payload = _mock_payload(
            sequence=sequence,
            model_id=args.model_id,
            note=f"mock fallback: {exc.__class__.__name__}",
        )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
