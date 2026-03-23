from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import load_config, run_maple


def _phase_markdown(report: dict) -> str:
    observed = report.get("observed", {})
    thresholds = report.get("thresholds", {})
    unmet = report.get("phase3_unmet_gates", [])

    lines = [
        "# MAPLE Phase Gate Report",
        "",
        f"- current_phase: {report.get('current_phase')}",
        f"- transition_decision: {report.get('transition_decision')}",
        f"- phase3_ready: {report.get('phase3_ready')}",
        "",
        "## Observed",
    ]
    for key, value in observed.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Thresholds"])
    for key, value in thresholds.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Unmet Gates"])
    if unmet:
        lines.extend([f"- {item}" for item in unmet])
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MAPLE and generate phase transition gate report")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--structure-backend", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/phase_gate_report")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config = load_config(config_path)
    overrides = {
        "num_iterations": args.num_iterations,
        "structure_backend": args.structure_backend,
    }

    final_state, resolved, artifact_dir = run_maple(
        config=config,
        overrides=overrides,
        output_dir=args.output_dir,
    )
    phase_report = final_state.get("phase_report", resolved.get("phase", {}))

    out_dir = Path(artifact_dir)
    json_path = out_dir / "phase_gate_report.json"
    md_path = out_dir / "phase_gate_report.md"
    json_path.write_text(json.dumps(phase_report, indent=2), encoding="utf-8")
    md_path.write_text(_phase_markdown(phase_report), encoding="utf-8")

    print(f"Phase report written: {json_path}")
    print(f"Markdown report written: {md_path}")


if __name__ == "__main__":
    main()
