"""Optimization agent for evolutionary sequence improvement."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from models.embedding_model import RandomEmbeddingModel
from utils.bo import propose_bo_mutations
from utils.diversity import select_diverse_sequences
from utils.mutation import random_mutation
from utils.pareto import select_top_by_pareto


class OptimizationAgent:
    """Selects top candidates and generates next iteration sequences."""

    def __init__(self, random_seed: int = 19) -> None:
        self.random_seed = random_seed

    def _as_float(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _constraint_violations(self, prop: dict, structure: dict, config: dict) -> dict[str, float]:
        violations: dict[str, float] = {}

        min_stability = self._as_float(config.get("min_stability"))
        min_activity = self._as_float(config.get("min_activity"))
        min_structure_confidence = self._as_float(config.get("min_structure_confidence"))
        min_plddt = self._as_float(config.get("min_plddt"))
        min_ptm = self._as_float(config.get("min_ptm"))
        max_pae = self._as_float(config.get("max_pae"))

        stability = self._as_float(prop.get("stability"))
        activity = self._as_float(prop.get("activity"))
        confidence = self._as_float(structure.get("confidence"))
        plddt = self._as_float(structure.get("plddt_mean"))
        ptm = self._as_float(structure.get("ptm"))
        pae = self._as_float(structure.get("pae_mean"))

        if min_stability is not None and (stability is None or stability < min_stability):
            violations["min_stability"] = 1.0 if stability is None else float(min_stability - stability)
        if min_activity is not None and (activity is None or activity < min_activity):
            violations["min_activity"] = 1.0 if activity is None else float(min_activity - activity)
        if min_structure_confidence is not None and (confidence is None or confidence < min_structure_confidence):
            violations["min_structure_confidence"] = (
                1.0 if confidence is None else float(min_structure_confidence - confidence)
            )
        if min_plddt is not None and (plddt is None or plddt < min_plddt):
            violations["min_plddt"] = 1.0 if plddt is None else float(min_plddt - plddt)
        if min_ptm is not None and (ptm is None or ptm < min_ptm):
            violations["min_ptm"] = 1.0 if ptm is None else float(min_ptm - ptm)
        if max_pae is not None and (pae is None or pae > max_pae):
            violations["max_pae"] = 1.0 if pae is None else float(pae - max_pae)

        return violations

    def _objective_vector(self, prop: dict, structure: dict) -> np.ndarray:
        stability = self._as_float(prop.get("stability"))
        activity = self._as_float(prop.get("activity"))
        uncertainty = self._as_float(prop.get("uncertainty"))
        confidence = self._as_float(structure.get("confidence"))
        return np.asarray(
            [
                0.0 if stability is None else float(stability),
                0.0 if activity is None else float(activity),
                -1.0 * (0.0 if uncertainty is None else float(uncertainty)),
                0.0 if confidence is None else float(confidence),
            ],
            dtype=np.float32,
        )

    def _select_elites_pareto(self, candidate_records: list[dict], top_k: int, min_distance: int) -> list[str]:
        if not candidate_records:
            return []

        points = np.stack([self._objective_vector(rec["prop"], rec["structure"]) for rec in candidate_records])
        selected_idx = select_top_by_pareto(points, top_k=max(1, min(top_k, len(candidate_records))))
        selected = [candidate_records[i]["seq"] for i in selected_idx]

        if min_distance > 0 and len(selected) > 1:
            selected = select_diverse_sequences(selected, top_k=len(selected), min_distance=min_distance)
        return selected

    def _select_elites(self, ranked_sequences: list[str], top_k: int, strategy: str, min_distance: int) -> list[str]:
        if not ranked_sequences:
            return []

        if strategy == "diverse" and min_distance > 0:
            return select_diverse_sequences(ranked_sequences, top_k=top_k, min_distance=min_distance)

        return ranked_sequences[: max(1, min(top_k, len(ranked_sequences)))]

    def run(self, state: dict) -> dict:
        config = state.get("config", {})
        top_k = int(config.get("top_k", 3))
        num_candidates = int(config.get("num_candidates", 8))
        mutation_rate = int(config.get("mutation_rate", 1))
        min_distance = int(config.get("min_hamming_distance", 0))
        strategy = str(config.get("selection_strategy", "elitist")).strip().lower()
        bo_beta = float(config.get("bo_beta", 0.30))
        bo_trials_per_parent = int(config.get("bo_trials_per_parent", 8))
        constraint_enabled = bool(config.get("constraint_enabled", False))
        constraint_mode = str(config.get("constraint_mode", "hard")).strip().lower()
        constraint_penalty = float(config.get("constraint_penalty", 0.20))
        iteration = int(state.get("iteration", 0))

        sequences = state.get("sequences", [])
        structures = state.get("structures", [])
        embeddings = state.get("embeddings", [])
        properties = state.get("properties", [])
        scores = state.get("scores", [])

        if not sequences:
            state["next_sequences"] = []
            return state

        if len(embeddings) != len(sequences):
            embeddings = [None for _ in sequences]

        ranked_aligned = list(zip(sequences, structures, embeddings, properties, scores))
        passed: list[dict[str, Any]] = []
        all_ranked: list[dict[str, Any]] = []
        violation_counts: dict[str, int] = {
            "min_stability": 0,
            "min_activity": 0,
            "min_structure_confidence": 0,
            "min_plddt": 0,
            "min_ptm": 0,
            "max_pae": 0,
        }

        for seq, structure, embedding, prop, score in ranked_aligned:
            violations = self._constraint_violations(prop, structure, config) if constraint_enabled else {}
            for k in violations:
                violation_counts[k] += 1

            if not violations:
                passed.append(
                    {
                        "seq": seq,
                        "structure": structure,
                        "embedding": embedding,
                        "prop": prop,
                        "score": float(score),
                        "violations": violations,
                        "penalized": float(score),
                    }
                )

            total_violation = sum(violations.values())
            penalized = float(score) - constraint_penalty * total_violation
            all_ranked.append(
                {
                    "seq": seq,
                    "structure": structure,
                    "embedding": embedding,
                    "prop": prop,
                    "score": float(score),
                    "violations": violations,
                    "penalized": penalized,
                }
            )

        if constraint_enabled and constraint_mode == "hard":
            candidate_records = passed if passed else all_ranked
        elif constraint_enabled and constraint_mode == "soft":
            candidate_records = sorted(all_ranked, key=lambda x: float(x["penalized"]), reverse=True)
        else:
            candidate_records = all_ranked

        state["constraint_summary"] = {
            "enabled": constraint_enabled,
            "mode": constraint_mode,
            "penalty": constraint_penalty,
            "passed": len(passed),
            "total": len(sequences),
            "violation_counts": violation_counts,
        }

        if strategy in {"pareto", "pareto_bo"}:
            elites = self._select_elites_pareto(
                candidate_records=candidate_records,
                top_k=max(1, min(top_k, len(candidate_records))),
                min_distance=min_distance,
            )
        else:
            candidate_ranked_sequences = [rec["seq"] for rec in candidate_records]
            elites = self._select_elites(
                ranked_sequences=candidate_ranked_sequences,
                top_k=max(1, min(top_k, len(candidate_ranked_sequences))),
                strategy=strategy,
                min_distance=min_distance,
            )

        rng = random.Random(self.random_seed + iteration)
        next_sequences = list(elites)

        if strategy == "pareto_bo":
            first_valid = next((e for e in embeddings if e is not None), None)
            inferred_dim = int(np.asarray(first_valid).shape[0]) if first_valid is not None else 128
            embedder = RandomEmbeddingModel(embedding_dim=inferred_dim)
            valid_embed_rows = [np.asarray(e, dtype=np.float32) for e in embeddings if e is not None]
            valid_scores = [float(scores[i]) for i, e in enumerate(embeddings) if e is not None]
            train_embeddings = (
                np.stack(valid_embed_rows).astype(np.float32)
                if valid_embed_rows
                else np.empty((0, inferred_dim), dtype=np.float32)
            )
            bo_children = propose_bo_mutations(
                parents=elites,
                train_embeddings=train_embeddings,
                train_scores=valid_scores,
                embedding_model=embedder,
                num_to_generate=max(0, num_candidates - len(next_sequences)),
                mutation_rate=mutation_rate,
                rng=rng,
                beta=bo_beta,
                trials_per_parent=bo_trials_per_parent,
            )
            for child in bo_children:
                if len(next_sequences) >= num_candidates:
                    break
                if child not in next_sequences:
                    next_sequences.append(child)

        while len(next_sequences) < num_candidates:
            parent = rng.choice(elites)
            child = random_mutation(parent, num_mutations=mutation_rate, rng=rng)
            next_sequences.append(child)

        state["next_sequences"] = next_sequences
        return state
