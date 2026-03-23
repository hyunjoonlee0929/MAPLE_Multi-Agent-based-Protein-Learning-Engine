"""Entry point for running the MAPLE MVP pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from agents.evaluation_agent import EvaluationAgent
from agents.optimization_agent import OptimizationAgent
from agents.planner import PlannerAgent
from agents.property_agent import PropertyAgent
from agents.sequence_agent import SequenceAgent
from agents.structure_agent import StructureAgent
from core.pipeline import MaplePipeline, PipelineConfig
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



def main() -> None:
    setup_logging()
    logger = logging.getLogger("MAPLE")

    config = load_config(Path(__file__).parent / "config.yaml")

    seed_sequence = config["seed_sequence"]
    num_iterations = int(config["num_iterations"])
    runtime_cfg = config.get("runtime", {})
    embedding_dim = int(config.get("model", {}).get("embedding_dim", 128))

    state = create_initial_state(seed_sequence)
    state["config"] = runtime_cfg

    pipeline = MaplePipeline(
        config=PipelineConfig(num_iterations=num_iterations),
        planner_agent=PlannerAgent(),
        sequence_agent=SequenceAgent(),
        structure_agent=StructureAgent(),
        property_agent=PropertyAgent(embedding_dim=embedding_dim),
        optimization_agent=OptimizationAgent(),
        evaluation_agent=EvaluationAgent(),
        logger=logger,
    )

    final_state = pipeline.run(state)

    logger.info("Run completed.")
    logger.info("Best sequence: %s", final_state["sequences"][0] if final_state["sequences"] else None)
    logger.info("Best score: %s", final_state["scores"][0] if final_state["scores"] else None)
    logger.info("History entries: %d", len(final_state["history"]))

    print("\n=== MAPLE Summary ===")
    print(f"Project: {config.get('project_title')}")
    print(f"Iterations: {num_iterations}")
    print(f"Final best sequence: {final_state['sequences'][0] if final_state['sequences'] else 'N/A'}")
    print(f"Final best score: {final_state['scores'][0] if final_state['scores'] else 'N/A'}")


if __name__ == "__main__":
    main()
