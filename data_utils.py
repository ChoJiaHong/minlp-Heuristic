"""Utility functions for loading and generating heuristic datasets."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence


Dataset = Dict[str, object]


def _weighted_sample_without_replacement(
    population: Sequence[str], weights: Sequence[float], k: int, *, rng: random.Random
) -> List[str]:
    """Sample ``k`` unique elements using the provided weights.

    ``random.choices`` does not support sampling without replacement, so this helper
    performs a simple rejection-sampling loop with a running total of remaining weight.
    The population size in this project is small (services are < 100), so the
    straightforward approach keeps the code easy to read.
    """

    if k <= 0:
        return []
    if k >= len(population):
        return list(population)

    remaining_items = list(population)
    remaining_weights = list(weights)
    selected: List[str] = []

    for _ in range(k):
        total = sum(remaining_weights)
        threshold = rng.random() * total
        cumulative = 0.0
        for idx, (item, weight) in enumerate(zip(remaining_items, remaining_weights)):
            cumulative += weight
            if cumulative >= threshold:
                selected.append(item)
                remaining_items.pop(idx)
                remaining_weights.pop(idx)
                break
    return selected


def _maybe_drop_dominated(
    combos: MutableMapping[str, Mapping[str, object]],
    candidate_services: Iterable[str],
    candidate_weights: Mapping[str, int],
) -> bool:
    """Drop dominated combinations and report if the candidate is dominated.

    A combination is *dominated* when another combination provides the same service
    set with strictly lower or equal weights for every service. The function removes
    dominated existing entries to keep the dataset compact.
    """

    candidate_set = set(candidate_services)
    dominated = False
    to_remove = []

    for name, combo in combos.items():
        services = set(combo["services"])
        weights = combo["W"]  # type: ignore[assignment]
        if services != candidate_set:
            continue

        if all(candidate_weights[q] >= weights[q] for q in services):
            # Candidate is dominated by an existing combo
            dominated = True
            break

        if all(weights[q] >= candidate_weights[q] for q in services):
            # Existing combo is dominated by the candidate
            to_remove.append(name)

    for name in to_remove:
        combos.pop(name)
    return dominated


def generate_random_data(config: Mapping[str, object]) -> Dataset:
    """Generate a random dataset that mirrors ``random_data.json`` structure."""

    rng = random.Random(config.get("seed", None))
    num_nodes = int(config["num_nodes"])
    num_services = int(config["num_services"])
    num_agents = int(config["num_agents"])
    combos_per_node = int(config["combos_per_node"])
    avg_services_per_combo = float(config["avg_services_per_combo"])
    min_req = int(config["min_req_services"])
    max_req = int(config["max_req_services"])
    capacity_low = int(config["capacity_low"])
    capacity_high = int(config["capacity_high"])
    req_pattern = str(config.get("req_pattern", "random"))
    drop_dominated = bool(config.get("drop_dominated", True))
    fl_ratio = float(config.get("fl_ratio", 0.15))
    fh_ratio = float(config.get("fh_ratio", 0.30))

    M = [f"m{i+1}" for i in range(num_nodes)]
    Q = [f"q{i+1}" for i in range(num_services)]
    G = [f"g{i+1}" for i in range(num_agents)]

    base_caps = {q: rng.randint(capacity_low, capacity_high) for q in Q}
    f_l = {q: math.ceil(base_caps[q] * fl_ratio) for q in Q}
    f_h = {q: math.ceil(base_caps[q] * fh_ratio) for q in Q}

    combos = {}
    for m in M:
        combos[m] = {}
        counter = 1
        while len(combos[m]) < combos_per_node:
            size = max(1, min(len(Q), int(round(rng.gauss(avg_services_per_combo, 1.2)))))
            services = rng.sample(Q, size)
            weights = {q: rng.randint(1, base_caps[q]) for q in services}
            name = f"c{counter}"
            if drop_dominated and _maybe_drop_dominated(combos[m], services, weights):
                continue
            combos[m][name] = {"services": services, "W": weights}
            counter += 1

    def build_requirements() -> Dict[str, List[str]]:
        requirements: Dict[str, List[str]] = {}
        weights = [len(Q) - idx for idx in range(len(Q))]
        for idx, g in enumerate(G):
            req_size = rng.randint(min_req, max_req)
            if req_pattern == "cycle":
                start = idx % len(Q)
                ordered = Q[start:] + Q[:start]
                requirements[g] = ordered[:req_size]
            elif req_pattern == "heavy_first":
                requirements[g] = _weighted_sample_without_replacement(Q, weights, req_size, rng=rng)
            else:
                requirements[g] = rng.sample(Q, req_size)
        return requirements

    R = build_requirements()

    dataset: Dataset = {
        "M": M,
        "Q": Q,
        "G": G,
        "R": R,
        "combos": combos,
        "f_l": f_l,
        "f_h": f_h,
        "base_caps": base_caps,
    }
    return dataset


def example_data() -> Dataset:
    """Return a tiny hard-coded dataset useful for quick sanity checks."""

    return {
        "M": ["m1", "m2"],
        "Q": ["q1", "q2"],
        "G": ["g1", "g2", "g3"],
        "R": {"g1": ["q1"], "g2": ["q1", "q2"], "g3": ["q2"]},
        "combos": {
            "m1": {
                "c1": {"services": ["q1"], "W": {"q1": 3}},
                "c2": {"services": ["q1", "q2"], "W": {"q1": 2, "q2": 1}},
            },
            "m2": {"c1": {"services": ["q2"], "W": {"q2": 2}}},
        },
        "f_l": {"q1": 1, "q2": 1},
        "f_h": {"q1": 2, "q2": 2},
        "base_caps": {"q1": 5, "q2": 3},
    }


def save_dataset(dataset: Dataset, path: str | Path) -> None:
    Path(path).write_text(json.dumps(dataset, indent=2))


def load_dataset(mode: str, config: Mapping[str, object]) -> Dataset:
    """Load a dataset according to the configuration mode."""

    if mode == "example":
        return example_data()
    if mode == "random":
        random_cfg = config.get("random")
        if not isinstance(random_cfg, Mapping):
            raise ValueError("DATA_CONFIG['random'] must be a mapping when mode=='random'")
        return generate_random_data(random_cfg)
    if mode == "json":
        path = Path(str(config["json_path"]))
        return json.loads(path.read_text())
    raise ValueError(f"Unsupported mode: {mode}")


def greedy_assignment(dataset: Mapping[str, object]) -> Dict[str, object]:
    """Assign generators to node combos greedily based on capacity usage."""

    M: List[str] = dataset["M"]  # type: ignore[assignment]
    combos: Mapping[str, Mapping[str, Mapping[str, object]]] = dataset["combos"]  # type: ignore[assignment]
    requirements: Mapping[str, List[str]] = dataset["R"]  # type: ignore[assignment]
    capacities: Dict[str, int] = dict(dataset["base_caps"])  # type: ignore[arg-type]

    assignments: Dict[str, Dict[str, str]] = {}
    unassigned: List[str] = []

    for g in dataset["G"]:  # type: ignore[assignment]
        reqs = set(requirements[g])
        best_choice = None
        best_weight = None

        for m in M:
            for combo_name, combo in combos[m].items():
                services = set(combo["services"])
                if not reqs.issubset(services):
                    continue
                weights: Mapping[str, int] = combo["W"]  # type: ignore[assignment]
                if any(capacities[q] < weights[q] for q in reqs):
                    continue
                total_weight = sum(weights[q] for q in reqs)
                if best_weight is None or total_weight < best_weight:
                    best_choice = (m, combo_name, weights)
                    best_weight = total_weight

        if best_choice is None:
            unassigned.append(g)
            continue

        m, combo_name, weights = best_choice
        assignments[g] = {"node": m, "combo": combo_name}
        for q in reqs:
            capacities[q] -= weights[q]

    return {
        "assignments": assignments,
        "unassigned": unassigned,
        "remaining_caps": capacities,
    }
