"""
Utility functions for managing experiment configurations and data paths.
"""

import json
import os
from argparse import Namespace

from strenum import StrEnum

BASE_EXPERIMENT_DATA_PATH = "./eval/experiment_data/"


class RLAlgorithm(StrEnum):
    """
    Enum representing different algorithms used in experiments.
    """

    RLOO = "rloo"
    OUTCOME_GRPO = "outcome_grpo"
    PROCESS_GRPO = "process_grpo"  # Process GRPO according to Emanuel Ruzak (https://github.com/emparu/PPO-vs-GRPO)
    DEEPSEEK_PROCESS_GRPO = "deepseek_process_grpo"  # Process GRPO according to DeepSeek (https://arxiv.org/pdf/2402.03300)
    PPO = "ppo"


def get_experiment_data_path(experiment_id: int, algorithm: str | None = None, run_id: int | None = None) -> str:
    """
    Returns the path to the experiment data directory for an experiment ID and optionally for a given algorithm and run id.
    If the run_id is provided, it will return the path to that specific run.
    """
    if algorithm is None:
        return os.path.join(BASE_EXPERIMENT_DATA_PATH, f"experiment_{experiment_id}")
    if run_id is None:
        return os.path.join(BASE_EXPERIMENT_DATA_PATH, f"experiment_{experiment_id}", algorithm)
    return os.path.join(BASE_EXPERIMENT_DATA_PATH, f"experiment_{experiment_id}", algorithm, f"run_{run_id}")


def get_existing_runs(experiment_id: int, algorithm: str) -> int:
    """
    Returns the number of existing runs for a given experiment ID and algorithm.
    """
    path = get_experiment_data_path(experiment_id, algorithm)
    if not os.path.exists(path):
        return 0
    return len([name for name in os.listdir(path) if name.startswith("run_")])


def save_experiment_config(args: Namespace):
    """
    Saves the experiment config to a JSON file in the experiment data directory.
    If the experiment data directory does not exist, it creates it.
    If the config file already exists, it checks for mismatches in non-permitted fields and merges algorithm lists.
    If there are mismatches in non-permitted fields, it raises a ValueError.
    """
    experiment_path = get_experiment_data_path(args.exp_id)
    os.makedirs(experiment_path, exist_ok=True)
    exp_config_path = os.path.join(experiment_path, "exp_config.json")
    new_config = vars(args).copy()

    if os.path.exists(exp_config_path):
        with open(exp_config_path, "r", encoding="utf-8") as f:
            existing_config = json.load(f)

            # Check that specifications are the same (except for alg and n-runs, these will be handled by experiment_eval.py)
            ignore_keys = {"alg", "n_runs"}
            mismatches = {
                key: (existing_config[key], new_config[key])
                for key in new_config
                if key not in ignore_keys and key in existing_config and existing_config[key] != new_config[key]
            }
            if mismatches:
                mismatch_str = "\n".join(f"  {k}: existing={v[0]}, new={v[1]}" for k, v in mismatches.items())
                raise ValueError(f"Config mismatch in non-permitted fields:\n{mismatch_str}")

            # Merge algorithm lists
            existing_algs = set(existing_config.get("alg", []))
            new_algs = set(new_config.get("alg", []))
            existing_config["alg"] = list(existing_algs | new_algs)
            new_config = existing_config

    with open(exp_config_path, "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=4)


# TODO write alg config files
