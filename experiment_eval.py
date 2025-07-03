"""
Evaluate and plot results of reinforcement learning experiments.
"""

import argparse
import json
import os
from datetime import datetime
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

import experiments.utils as exp_utils
from experiments.utils import RLAlgorithm

warn(
    "This script is legacy code that should not be used anymore. "
    "Please consult the guide at `EXPERIMENT_RLIABLE_USAGE.md` for help about "
    "how to use SB3's RL-Zoo3 for clean training runs and results plotting.",
    DeprecationWarning,
)

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

# Plot curves for each algorithm averaged over runs
plt.figure(figsize=(14, 5))
plt.suptitle(f"Experiment {args.exp_id} Evaluation over {n_runs} runs", fontsize=16)

if args.plot_timesteps:
    # Plot rewards vs timesteps
    plt.subplot(1, 2, 1)

    for alg in n_runs_dict.keys():
        # Collect all data for the algorithm across runs
        all_alg_data = []
        for i in range(n_runs):
            data = dict(np.load(exp_utils.get_experiment_data_path(args.exp_id, alg, i)))
            all_alg_data.append(data)

        all_timesteps = [data["timesteps"] for data in all_alg_data]
        common_timesteps = set(all_timesteps[0])
        for timesteps in all_timesteps[1:]:
            common_timesteps.intersection_update(timesteps)
        common_timesteps = sorted(common_timesteps)

        # Filter data to only include common timesteps between all runs
        dropped_timesteps = []
        for i, data in enumerate(all_alg_data):
            ts = data["timesteps"]
            results = data["results"]

            indices = [i for i, t in enumerate(ts) if t in common_timesteps]
            dropped_timesteps.append(len(ts) - len(indices))

            data["timesteps"] = ts[indices]
            data["results"] = results[indices]

        # Sanity check: all timesteps must be equal across runs
        for data in all_alg_data:
            a1 = data["timesteps"]
            a2 = all_alg_data[0]["timesteps"]
            assert np.array_equal(a1, a2), f"Timesteps must be equal: {a1} != {a2}"

        all_rewards = np.array([data["results"] for data in all_alg_data])
        all_rewards = np.mean(all_rewards, axis=2)
        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        plt.plot(common_timesteps, mean_rewards, label=f"{alg.name.upper()}")
        plt.fill_between(common_timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

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
            data = dict(np.load(exp_utils.get_experiment_data_path(args.exp_id, alg, i)))
            all_alg_data.append(data)

        # Only keep first update observations for each run
        for i, data in enumerate(all_alg_data):
            updates = data["updates"]
            results = data["results"]

            seen = set()
            indices = []
            for idx, upd in enumerate(updates):
                # IDEA check if I can do this with np.unique
                if upd not in seen:
                    indices.append(idx)
                    seen.add(upd)

            data["updates"] = updates[indices]
            data["results"] = results[indices]

        # Only keep common updates across all runs
        all_updates = [data["updates"] for data in all_alg_data]
        common_updates = set(all_updates[0])
        for updates in all_updates[1:]:
            common_updates.intersection_update(updates)
        common_updates = sorted(common_updates)

        # Filter data to only include common updates between all runs
        dropped_updates = []
        for i, data in enumerate(all_alg_data):
            updates = data["updates"]
            results = data["results"]

            indices = [i for i, upd in enumerate(updates) if upd in common_updates]
            dropped_updates.append(len(updates) - len(indices))

            data["updates"] = updates[indices]
            data["results"] = results[indices]

        # Sanity check that all updates are equal across runs
        for data in all_alg_data:
            a1 = data["updates"]
            a2 = all_alg_data[0]["updates"]
            assert np.array_equal(a1, a2), f"Updates must be equal: {a1} != {a2}"

        all_rewards = np.array([data["results"] for data in all_alg_data])
        all_rewards = np.mean(all_rewards, axis=2)
        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        plt.plot(common_updates, mean_rewards, label=f"{alg.name.upper()}")
        plt.fill_between(common_updates, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)

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
