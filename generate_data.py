"""Generate a dataset file based on ``config.DATA_CONFIG``."""

from __future__ import annotations

import argparse

from config import DATA_CONFIG
from data_utils import generate_random_data, save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a heuristic dataset")
    parser.add_argument(
        "--out",
        default=DATA_CONFIG.get("out", "random_data.json"),
        help="Output path for the generated JSON dataset",
    )
    args = parser.parse_args()

    dataset = generate_random_data(DATA_CONFIG["random"])
    save_dataset(dataset, args.out)
    print(f"Dataset written to {args.out}")


if __name__ == "__main__":
    main()
