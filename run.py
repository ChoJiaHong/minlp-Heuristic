"""Run a simple greedy heuristic using the dataset configuration."""

from __future__ import annotations

import argparse
from pprint import pprint

from config import DATA_CONFIG
from data_utils import greedy_assignment, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run greedy assignment on a dataset")
    parser.add_argument(
        "--mode",
        choices=["example", "json", "random"],
        default=DATA_CONFIG.get("mode", "json"),
        help="Which data source to use",
    )
    parser.add_argument(
        "--json-path",
        default=DATA_CONFIG.get("json_path", "random_data.json"),
        help="Path to dataset file when mode=json",
    )
    args = parser.parse_args()

    config = dict(DATA_CONFIG)
    if args.mode == "json":
        config["json_path"] = args.json_path

    dataset = load_dataset(args.mode, config)
    results = greedy_assignment(dataset)

    print("Assignment summary:")
    print(f"  Assigned generators: {len(results['assignments'])}")
    print(f"  Unassigned generators: {len(results['unassigned'])}")
    print("  Remaining capacities:")
    pprint(results["remaining_caps"])

    if results["unassigned"]:
        print("\nUnassigned generators:")
        print(", ".join(results["unassigned"]))


if __name__ == "__main__":
    main()
