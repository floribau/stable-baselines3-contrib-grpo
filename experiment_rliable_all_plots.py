"""
Module doing preprocessing and then calling RL-Zoo3 all_plots.py script
"""

import argparse
import os

import numpy as np

from rl_zoo3.plots.all_plots import all_plots


def experiment_rliable_all_plots():
    """
    Preprocesses the results of an experiment to ensure that all runs with the same
    experiment ID, environment, and algorithm have the same number of timesteps.
    Any excess timesteps are truncated.
    This is necessary because the RL Zoo 3 all_plots script expects grouped runs to have
    the same number of timesteps for plotting purposes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algos", help="Algorithms to include", nargs="+", type=str)
    parser.add_argument("-e", "--env", help="Environments to include", nargs="+", type=str)
    parser.add_argument("-f", "--exp-folders", help="Folders to include", nargs="+", type=str)

    args, _ = parser.parse_known_args()

    # Truncate results to the minimum number of timesteps across all runs with the same exp id, env and algo
    for folder in args.exp_folders:
        for algo in args.algos:
            for env in args.env:
                min_timesteps = float("inf")
                data_dict = {}
                algo_path = os.path.join(folder, algo)
                for d in os.listdir(algo_path):
                    if d.startswith(env):
                        data_path = os.path.join(algo_path, d, "evaluations.npz")
                        if os.path.exists(data_path):
                            data = dict(np.load(data_path))
                            data_dict[data_path] = data
                            min_timesteps = min(min_timesteps, data["timesteps"].shape[0])
                for data_path, data in data_dict.items():
                    for key in data.keys():
                        data[key] = data[key][:min_timesteps]
                    np.savez(data_path, **data)

    # Call the RL Zoo 3 all_plots script
    all_plots()


if __name__ == "__main__":
    experiment_rliable_all_plots()
