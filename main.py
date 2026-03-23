"""Entry point for running the MAPLE MVP pipeline."""

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
    """Set global seeds for reproducible MVP runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
    except ModuleNotFoundError:
        pass



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAPLE multi-agent protein optimization")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Global random seed")

    parser.add_argument("--seed-sequence", type=str, default=None, help="Override seed protein sequence")
    parser.add_argument("--num-iterations", type=int, default=None, help="Override iteration count")

    parser.add_argument("--num-candidates", type=int, default=None, help="Override candidates per iteration")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k elite count")
    parser.add_argument("--mutation-rate", type=int, default=None, help="Override mutations per sequence")

    parser.add_argument("--embedding-dim", type=int, default=None, help="Override embedding dimension")
    parser.add_argument(
        "--property-checkpoint",
        type=str,
        default=None,
        help="Optional PyTorch checkpoint path for property predictor",
    )
    parser.add_argument(
        "--structure-backend",
        type=str,
        default=None,
        help="Structure backend: dummy|esmfold|alphafold2",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for history and summary artifacts",
    )

    return parser.parse_args()



def main() -> None:
    setup_logging()
    logger = logging.getLogger("MAPLE")
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    config = load_config(config_path)

    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_global_seed(seed)

    seed_sequence = args.seed_sequence or config["seed_sequence"]
    num_iterations = int(args.num_iterations if args.num_iterations is not None else config["num_iterations"])

    runtime_cfg = dict(config.get("runtime", {}))
    if args.num_candidates is not None:
        runtime_cfg["num_candidates"] = args.num_candidates
    if args.top_k is not None:
        runtime_cfg["top_k"] = args.top_k
    if args.mutation_rate is not None:
        runtime_cfg["mutation_rate"] = args.mutation_rate

    model_cfg = dict(config.get("model", {}))
    embedding_dim = int(
        args.embedding_dim if args.embedding_dim is not None else model_cfg.get("embedding_dim", 128)
    )
    property_checkpoint = args.property_checkpoint or model_cfg.get("property_checkpoint")
    structure_backend = args.structure_backend or model_cfg.get("structure_backend", "dummy")

    state = create_initial_state(seed_sequence)
    state["config"] = runtime_cfg

    pipeline = MaplePipeline(
        config=PipelineConfig(num_iterations=num_iterations),
        planner_agent=PlannerAgent(),
        sequence_agent=SequenceAgent(random_seed=seed + 11),
        structure_agent=StructureAgent(backend=structure_backend),
        property_agent=PropertyAgent(
            embedding_dim=embedding_dim,
            property_checkpoint=property_checkpoint,
        ),
        optimization_agent=OptimizationAgent(random_seed=seed + 29),
        evaluation_agent=EvaluationAgent(),
        logger=logger,
    )

    final_state = pipeline.run(state)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / output_dir

    export_history_json(final_state["history"], output_dir / "history.json")
    export_history_csv(final_state["history"], output_dir / "history.csv")
    export_final_summary(final_state, output_dir / "summary.json")

    logger.info("Run completed.")
    logger.info("Best sequence: %s", final_state["sequences"][0] if final_state["sequences"] else None)
    logger.info("Best score: %s", final_state["scores"][0] if final_state["scores"] else None)
    logger.info("History entries: %d", len(final_state["history"]))
    logger.info("Artifacts written to: %s", output_dir)

    print("\n=== MAPLE Summary ===")
    print(f"Project: {config.get('project_title')}")
    print(f"Iterations: {num_iterations}")
    print(f"Seed: {seed}")
    print(f"Structure backend: {structure_backend}")
    print(f"Final best sequence: {final_state['sequences'][0] if final_state['sequences'] else 'N/A'}")
    print(f"Final best score: {final_state['scores'][0] if final_state['scores'] else 'N/A'}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
