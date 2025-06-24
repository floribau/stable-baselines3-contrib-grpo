"""
Evaluate and plot results of reinforcement learning experiments.
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import experiments.utils as exp_utils
from experiments.utils import RLAlgorithm

parser = argparse.ArgumentParser()
parser.add_argument("--exp-id", type=int, required=True, help="Experiment ID")
parser.add_argument("--plot-timesteps", action="store_true", help="Plot rewards vs timesteps")
parser.add_argument("--plot-updates", action="store_true", help="Plot rewards vs updates")
args = parser.parse_args()


# Get minimum number of runs of all algorithms in the experiment
n_runs_dict = {
    alg: exp_utils.get_existing_runs(experiment_id=args.exp_id, algorithm=alg) for alg in RLAlgorithm.__members__.values()
}
n_runs_dict = {alg: n_runs for alg, n_runs in n_runs_dict.items() if n_runs > 0}
n_runs = min(n_runs_dict.values())

exp_path = exp_utils.get_experiment_data_path(args.exp_id)
with open(os.path.join(exp_path, f"exp_{args.exp_id}_config.json"), "r", encoding="utf-8") as f:
    exp_config = json.load(f)
num_timesteps = exp_config.get("n_training_steps", 0)
eval_freq = exp_config.get("eval_freq", 0)
num_common_steps = max(int(num_timesteps / eval_freq), 1)  # TODO does this work for plot_updates as well?

# Plot curves for each algorithm averaged over runs
plt.figure(figsize=(14, 5))
plt.suptitle(f"Experiment {args.exp_id} Evaluation", fontsize=16)

if args.plot_timesteps:
    # Plot rewards vs timesteps
    plt.subplot(1, 2, 1)

    for alg in n_runs_dict.keys():
        # Collect all data for the algorithm across runs
        all_alg_data = []
        for i in range(n_runs):
            data = np.load(exp_utils.get_experiment_data_path(args.exp_id, alg, i))
            all_alg_data.append(data)

        min_ts = max(data["timesteps"][0] for data in all_alg_data)
        max_ts = min(data["timesteps"][-1] for data in all_alg_data)
        common_steps = np.linspace(min_ts, max_ts, num_common_steps)

        for data in all_alg_data:
            for key in data.keys():
                arr = data[key]
                print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
                print(f"{key} sample values: {arr[:5]}")

        all_interpolated_rewards = []
        for data in all_alg_data:
            timesteps = data["timesteps"]
            rewards = data["results"]
            mean_rewards = np.mean(rewards, axis=1)

            # Interpolate rewards to common steps
            interp_rewards = np.interp(common_steps, timesteps, mean_rewards)
            all_interpolated_rewards.append(interp_rewards)

        all_interpolated_rewards = np.array(all_interpolated_rewards)
        mean_interp_rewards = np.mean(all_interpolated_rewards, axis=0)
        std_interp_rewards = np.std(all_interpolated_rewards, axis=0)

        plt.plot(common_steps, mean_interp_rewards, label=f"{alg.name.upper()}")
        plt.fill_between(
            common_steps, mean_interp_rewards - std_interp_rewards, mean_interp_rewards + std_interp_rewards, alpha=0.2
        )

    plt.title("Rewards vs Timesteps")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward across Runs")
    plt.legend(loc="best")
    plt.grid(True)


if args.plot_updates:
    # Plot rewards vs updates
    plt.subplot(1, 2, 2)

    for alg in n_runs_dict.keys():
        # Collect all data for the algorithm across runs
        all_alg_data = []
        for i in range(n_runs):
            data = np.load(exp_utils.get_experiment_data_path(args.exp_id, alg, i))
            all_alg_data.append(data)

        min_upd = max((data["updates"][0] for data in all_alg_data), default=0)
        max_upd = min((data["updates"][-1] for data in all_alg_data), default=0)
        common_steps = np.linspace(min_upd, max_upd, num_common_steps)

        all_interpolated_rewards = []
        for data in all_alg_data:
            updates = data["updates"]
            rewards = data["results"]
            mean_rewards = np.mean(rewards, axis=1)

            # Interpolate rewards to common steps
            interp_rewards = np.interp(common_steps, updates, mean_rewards)
            all_interpolated_rewards.append(interp_rewards)

        all_interpolated_rewards = np.array(all_interpolated_rewards)
        mean_interp_rewards = np.mean(all_interpolated_rewards, axis=0)
        std_interp_rewards = np.std(all_interpolated_rewards, axis=0)

        plt.plot(common_steps, mean_interp_rewards, label=f"{alg.name.upper()}")
        plt.fill_between(
            common_steps, mean_interp_rewards - std_interp_rewards, mean_interp_rewards + std_interp_rewards, alpha=0.2
        )

    plt.title("Rewards vs Updates")
    plt.xlabel("Updates")
    plt.ylabel("Mean Reward across Runs")
    plt.legend(loc="best")
    plt.grid(True)

# Show and save the plot
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
plt.savefig(os.path.join(exp_path, f"experiment_{args.exp_id}_eval_{timestamp}.png"))
plt.show()
