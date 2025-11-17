# MINLP Heuristic Toolkit

This repository provides a small toolkit for generating synthetic datasets and running a greedy heuristic on them.

## Files
- `config.py`: Central configuration for data generation and run settings.
- `data_utils.py`: Shared helpers to generate/load datasets and perform greedy assignments.
- `generate_data.py`: CLI to generate a random dataset JSON file using the `config.py` parameters.
- `run.py`: CLI that loads data (example, JSON, or freshly generated) and executes the greedy assignment.

## Usage

### Generate a dataset
```bash
python generate_data.py --out random_data.json
```

### Run the greedy heuristic
Load the dataset according to the configured mode (see `config.py`). For example, using the JSON dataset saved above:
```bash
python run.py --mode json --json-path random_data.json
```

To quickly sanity-check the algorithm with a built-in small dataset:
```bash
python run.py --mode example
```
