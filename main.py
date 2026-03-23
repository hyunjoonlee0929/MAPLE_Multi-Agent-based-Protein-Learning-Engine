"""Entry point and run service for MAPLE."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np

from agents.evaluation_agent import EvaluationAgent
from agents.optimization_agent import OptimizationAgent
from agents.planner import PlannerAgent
from agents.property_agent import PropertyAgent
from agents.sequence_agent import SequenceAgent
from agents.structure_agent import StructureAgent
from core.pipeline import MaplePipeline, PipelineConfig
from core.reporting import export_final_summary, export_history_csv, export_history_json
from core.state import create_initial_state



def _coerce_yaml_scalar(value: str):
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value



def _simple_yaml_load(text: str) -> dict:
    """Very small YAML subset loader for this MVP config shape."""
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if value == "":
            new_obj: dict = {}
            current[key] = new_obj
            stack.append((indent, new_obj))
        else:
            current[key] = _coerce_yaml_scalar(value)

    return root



def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        content = f.read()

    try:
        import yaml  # type: ignore

        return yaml.safe_load(content)
    except ModuleNotFoundError:
        return _simple_yaml_load(content)



def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )



def set_global_seed(seed: int) -> None:
    """Set global seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
    except ModuleNotFoundError:
        pass



def run_maple(
    config: dict,
    overrides: dict | None = None,
    output_dir: str | Path = "outputs",
    logger: logging.Logger | None = None,
) -> tuple[dict, dict, Path]:
    """Run MAPLE with resolved config and optional overrides.

    Returns:
      final_state, resolved_runtime_info, resolved_output_dir
    """
    logger = logger or logging.getLogger("MAPLE")
    overrides = overrides or {}

    def _pick(name: str, default):
        value = overrides.get(name)
        return default if value is None else value

    seed = int(_pick("seed", config.get("seed", 42)))
    set_global_seed(seed)

    seed_sequence = str(_pick("seed_sequence", config["seed_sequence"]))
    num_iterations = int(_pick("num_iterations", config["num_iterations"]))

    runtime_cfg = dict(config.get("runtime", {}))
    for key in [
        "num_candidates",
        "top_k",
        "mutation_rate",
        "selection_strategy",
        "min_hamming_distance",
        "w_stability",
        "w_activity",
        "w_uncertainty",
        "w_structure",
    ]:
        if key in overrides and overrides[key] is not None:
            runtime_cfg[key] = overrides[key]

    model_cfg = dict(config.get("model", {}))
    embedding_dim = int(_pick("embedding_dim", model_cfg.get("embedding_dim", 128)))
    property_checkpoint = _pick("property_checkpoint", model_cfg.get("property_checkpoint"))
    structure_backend = str(_pick("structure_backend", model_cfg.get("structure_backend", "dummy")))
    uncertainty_samples = int(_pick("uncertainty_samples", model_cfg.get("uncertainty_samples", 5)))
    uncertainty_noise = float(_pick("uncertainty_noise", model_cfg.get("uncertainty_noise", 0.02)))
    structure_options = {
        "esmfold_command": _pick("esmfold_command", model_cfg.get("esmfold_command")),
        "alphafold2_command": _pick("alphafold2_command", model_cfg.get("alphafold2_command")),
        "structure_timeout_sec": _pick("structure_timeout_sec", model_cfg.get("structure_timeout_sec", 60)),
        "structure_retries": _pick("structure_retries", model_cfg.get("structure_retries", 0)),
    }
    structure_batch_size = int(_pick("structure_batch_size", model_cfg.get("structure_batch_size", 16)))

    state = create_initial_state(seed_sequence)
    state["config"] = runtime_cfg

    pipeline = MaplePipeline(
        config=PipelineConfig(num_iterations=num_iterations),
        planner_agent=PlannerAgent(),
        sequence_agent=SequenceAgent(random_seed=seed + 11),
        structure_agent=StructureAgent(
            backend=structure_backend,
            options=structure_options,
            batch_size=structure_batch_size,
        ),
        property_agent=PropertyAgent(
            embedding_dim=embedding_dim,
            property_checkpoint=property_checkpoint,
            uncertainty_samples=uncertainty_samples,
            uncertainty_noise=uncertainty_noise,
        ),
        optimization_agent=OptimizationAgent(random_seed=seed + 29),
        evaluation_agent=EvaluationAgent(),
        logger=logger,
    )

    final_state = pipeline.run(state)

    resolved_output_dir = Path(output_dir)
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = Path(__file__).parent / resolved_output_dir

    export_history_json(final_state["history"], resolved_output_dir / "history.json")
    export_history_csv(final_state["history"], resolved_output_dir / "history.csv")
    export_final_summary(final_state, resolved_output_dir / "summary.json")

    resolved = {
        "seed": seed,
        "num_iterations": num_iterations,
        "structure_backend": structure_backend,
        "selection_strategy": runtime_cfg.get("selection_strategy", "elitist"),
        "output_dir": str(resolved_output_dir),
    }
    return final_state, resolved, resolved_output_dir



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAPLE multi-agent protein optimization")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")

    parser.add_argument("--seed-sequence", type=str, default=None, help="Override seed protein sequence")
    parser.add_argument("--num-iterations", type=int, default=None, help="Override iteration count")

    parser.add_argument("--num-candidates", type=int, default=None, help="Override candidates per iteration")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k elite count")
    parser.add_argument("--mutation-rate", type=int, default=None, help="Override mutations per sequence")
    parser.add_argument("--selection-strategy", type=str, default=None, help="Elite selection strategy")
    parser.add_argument("--min-hamming-distance", type=int, default=None, help="Minimum Hamming distance")

    parser.add_argument("--w-stability", type=float, default=None, help="Score weight for stability")
    parser.add_argument("--w-activity", type=float, default=None, help="Score weight for activity")
    parser.add_argument("--w-uncertainty", type=float, default=None, help="Score weight for uncertainty")
    parser.add_argument("--w-structure", type=float, default=None, help="Score weight for structure confidence")

    parser.add_argument("--embedding-dim", type=int, default=None, help="Override embedding dimension")
    parser.add_argument("--property-checkpoint", type=str, default=None, help="Property checkpoint path")
    parser.add_argument("--uncertainty-samples", type=int, default=None, help="MC sample count")
    parser.add_argument("--uncertainty-noise", type=float, default=None, help="Input noise std")
    parser.add_argument("--structure-backend", type=str, default=None, help="Structure backend")
    parser.add_argument("--esmfold-command", type=str, default=None, help="Optional external ESMFold command")
    parser.add_argument("--alphafold2-command", type=str, default=None, help="Optional external AlphaFold2 command")
    parser.add_argument("--structure-timeout-sec", type=int, default=None, help="Timeout per structure external call")
    parser.add_argument("--structure-retries", type=int, default=None, help="Retry count for structure external calls")
    parser.add_argument("--structure-batch-size", type=int, default=None, help="Batch size for structure prediction loop")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Artifact directory")

    return parser.parse_args()



def main() -> None:
    setup_logging()
    logger = logging.getLogger("MAPLE")
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    config = load_config(config_path)

    overrides = {
        "seed": args.seed,
        "seed_sequence": args.seed_sequence,
        "num_iterations": args.num_iterations,
        "num_candidates": args.num_candidates,
        "top_k": args.top_k,
        "mutation_rate": args.mutation_rate,
        "selection_strategy": args.selection_strategy,
        "min_hamming_distance": args.min_hamming_distance,
        "w_stability": args.w_stability,
        "w_activity": args.w_activity,
        "w_uncertainty": args.w_uncertainty,
        "w_structure": args.w_structure,
        "embedding_dim": args.embedding_dim,
        "property_checkpoint": args.property_checkpoint,
        "uncertainty_samples": args.uncertainty_samples,
        "uncertainty_noise": args.uncertainty_noise,
        "structure_backend": args.structure_backend,
        "esmfold_command": args.esmfold_command,
        "alphafold2_command": args.alphafold2_command,
        "structure_timeout_sec": args.structure_timeout_sec,
        "structure_retries": args.structure_retries,
        "structure_batch_size": args.structure_batch_size,
    }

    final_state, resolved, output_dir = run_maple(
        config=config,
        overrides=overrides,
        output_dir=args.output_dir,
        logger=logger,
    )

    logger.info("Run completed.")
    logger.info("Best sequence: %s", final_state["sequences"][0] if final_state["sequences"] else None)
    logger.info("Best score: %s", final_state["scores"][0] if final_state["scores"] else None)
    logger.info("History entries: %d", len(final_state["history"]))
    logger.info("Artifacts written to: %s", output_dir)

    print("\n=== MAPLE Summary ===")
    print(f"Project: {config.get('project_title')}")
    print(f"Iterations: {resolved['num_iterations']}")
    print(f"Seed: {resolved['seed']}")
    print(f"Structure backend: {resolved['structure_backend']}")
    print(f"Selection strategy: {resolved['selection_strategy']}")
    print(f"Final best sequence: {final_state['sequences'][0] if final_state['sequences'] else 'N/A'}")
    print(f"Final best score: {final_state['scores'][0] if final_state['scores'] else 'N/A'}")
    print(f"Artifacts: {resolved['output_dir']}")


if __name__ == "__main__":
    main()
