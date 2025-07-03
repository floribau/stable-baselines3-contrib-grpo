"""Experiment module."""

import argparse
import os
import time
from warnings import warn

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.ppo import PPO

import experiments.utils as exp_utils
from experiments.utils import RLAlgorithm
from sb3_contrib.common.buffers import (
    DeepSeekOutcomeGroupBuffer,
    DeepSeekProcessGroupBuffer,
    ProcessGroupBuffer,
    SupervisionType,
)
from sb3_contrib.common.callbacks import UpdatesEvalCallback
from sb3_contrib.grpo.grpo import GRPO
from sb3_contrib.rloo.rloo import RLOO

warn(
    "This script is legacy code that should not be used anymore. "
    "Please refer to the guide at `EXPERIMENT_RLIABLE_USAGE.md` for help about "
    "how to use SB3's RL-Zoo3 for clean training runs and results plotting.",
    DeprecationWarning,
)

parser = argparse.ArgumentParser()
parser.add_argument("--exp-id", type=int, required=True, help="Experiment ID")
parser.add_argument(
    "--alg",
    type=str,
    required=True,
    action="append",
    choices=list(map(str, RLAlgorithm)),
    help="Algorithm(s) to use. Can be passed multiple times, e.g., --alg rloo --alg ppo",
)
parser.add_argument("--env", type=str, required=True, help="Environment name")
parser.add_argument("--normalize-env", action="store_true", help="Whether to normalize the environment")
parser.add_argument("--n-runs", type=int, default=1, help="Number of runs for the experiment")
parser.add_argument("--n-timesteps", type=int, default=100_000, help="Number of training steps per run")
parser.add_argument("--eval-freq", type=int, default=1_000, help="Frequency of evaluation during training in steps")
parser.add_argument("--n-eval-episodes", type=int, default=4, help="Number of episodes for evaluation")
parser.add_argument(
    "--verbose", type=int, default=0, choices=[0, 1, 2], help="Verbosity level (0: no output, 1: info, 2: debug)"
)
parser.add_argument("--collect-updates", action="store_true", help="Whether to also collect updates or only timesteps")
args = parser.parse_args()

EXPERIMENT_ID = args.exp_id
ENV_NAME = args.env
NORMALIZE_ENV = args.normalize_env
N_TRAINING_TIMESTEPS = args.n_timesteps
EVAL_FREQ = args.eval_freq
N_EVAL_EPISODES = args.n_eval_episodes
COLLECT_UPDATES = args.collect_updates

EXPERIMENT_PATH = exp_utils.get_experiment_data_path(EXPERIMENT_ID)
exp_utils.save_experiment_config(args)

# --- Training and Evaluation ---
# --- RLOO ---
if RLAlgorithm.RLOO in args.alg:
    # Vanilla RLOO with outcome supervision
    algorithm_name = RLAlgorithm.RLOO
    n_existing_runs = exp_utils.get_existing_runs(EXPERIMENT_ID, algorithm_name)
    alg_dir_path = exp_utils.get_experiment_data_path(EXPERIMENT_ID, algorithm_name)
    os.makedirs(alg_dir_path, exist_ok=True)

    # These specified model kwargs will only be used if no algorithm-specific config has been stored yet for the experiment
    model_kwargs = {
        "policy": "GroupPolicy",
        "group_size": 16,
        "kl_beta": 0,  # no KL penalty in standard RLOO
    }
    model_kwargs = exp_utils.get_algorithm_config(EXPERIMENT_ID, algorithm_name, model_kwargs)

    for run in range(n_existing_runs, args.n_runs + n_existing_runs):
        vec_env = make_vec_env(ENV_NAME, n_envs=1)
        eval_env = make_vec_env(ENV_NAME, n_envs=1)

        if NORMALIZE_ENV:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

        if COLLECT_UPDATES:
            eval_callback = UpdatesEvalCallback(
                eval_env,
                log_path=alg_dir_path,
                eval_freq=EVAL_FREQ,
                n_eval_episodes=N_EVAL_EPISODES,
                deterministic=True,
                render=False,
                verbose=0,
            )
        else:
            eval_callback = EvalCallback(
                eval_env,
                log_path=alg_dir_path,
                eval_freq=EVAL_FREQ,
                n_eval_episodes=N_EVAL_EPISODES,
                deterministic=True,
                render=False,
                verbose=0,
            )
        model = RLOO(
            env=vec_env,
            group_rollout_buffer_class=DeepSeekOutcomeGroupBuffer,
            verbose=args.verbose,
            **model_kwargs,
        )

        print(f"Starting Outcome RLOO training run {run}...")
        start_time = time.time()

        model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=eval_callback)
        model.save(os.path.join(alg_dir_path, f"model_{run}"))
        data = np.load(os.path.join(alg_dir_path, "evaluations.npz"))
        np.savez(exp_utils.get_experiment_data_path(args.exp_id, algorithm_name, run), **data)

        end_time = time.time()
        print(f"Outcome RLOO training run {run} completed in {(end_time - start_time):.2f} seconds.")

# --- GRPO ---
if RLAlgorithm.PROCESS_GRPO in args.alg:
    # Process supevision GRPO
    algorithm_name = RLAlgorithm.PROCESS_GRPO
    n_existing_runs = exp_utils.get_existing_runs(EXPERIMENT_ID, algorithm_name)
    alg_dir_path = exp_utils.get_experiment_data_path(EXPERIMENT_ID, algorithm_name)
    os.makedirs(alg_dir_path, exist_ok=True)

    # These specified model kwargs will only be used if no algorithm-specific config has been stored yet for the experiment
    model_kwargs = {
        "policy": "GroupPolicy",
        "group_size": 16,
        "n_epochs": 10,
    }
    model_kwargs = exp_utils.get_algorithm_config(EXPERIMENT_ID, algorithm_name, model_kwargs)

    for run in range(n_existing_runs, args.n_runs + n_existing_runs):
        vec_env = make_vec_env(ENV_NAME, n_envs=1)
        eval_env = make_vec_env(ENV_NAME, n_envs=1)
        if NORMALIZE_ENV:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

        if COLLECT_UPDATES:
            eval_callback = UpdatesEvalCallback(
                eval_env,
                log_path=alg_dir_path,
                eval_freq=EVAL_FREQ,
                n_eval_episodes=N_EVAL_EPISODES,
                deterministic=True,
                render=False,
                verbose=0,
            )
        else:
            eval_callback = EvalCallback(
                eval_env,
                log_path=alg_dir_path,
                eval_freq=EVAL_FREQ,
                n_eval_episodes=N_EVAL_EPISODES,
                deterministic=True,
                render=False,
                verbose=0,
            )
        model = GRPO(
            env=vec_env,
            group_rollout_buffer_class=ProcessGroupBuffer,
            verbose=args.verbose,
            **model_kwargs,
        )

        print(f"Starting Process GRPO training run {run}...")
        start_time = time.time()

        model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=eval_callback)
        model.save(os.path.join(alg_dir_path, f"model_{run}"))
        data = np.load(os.path.join(alg_dir_path, "evaluations.npz"))
        np.savez(exp_utils.get_experiment_data_path(args.exp_id, algorithm_name, run), **data)

        end_time = time.time()
        print(f"Process GRPO training run {run} completed in {(end_time - start_time):.2f} seconds.")

if RLAlgorithm.DEEPSEEK_PROCESS_GRPO in args.alg:
    # Process supevision GRPO conforming to DeepSeekMath paper
    algorithm_name = RLAlgorithm.DEEPSEEK_PROCESS_GRPO
    n_existing_runs = exp_utils.get_existing_runs(args.exp_id, algorithm_name)
    alg_dir_path = exp_utils.get_experiment_data_path(args.exp_id, algorithm_name)
    os.makedirs(alg_dir_path, exist_ok=True)

    # These specified model kwargs will only be used if no algorithm-specific config has been stored yet for the experiment
    model_kwargs = {
        "policy": "GroupPolicy",
        "group_size": 16,
        "n_epochs": 10,
    }
    model_kwargs = exp_utils.get_algorithm_config(args.exp_id, algorithm_name, model_kwargs)

    for run in range(n_existing_runs, args.n_runs + n_existing_runs):
        vec_env = make_vec_env(ENV_NAME, n_envs=1)
        eval_env = make_vec_env(ENV_NAME, n_envs=1)
        if NORMALIZE_ENV:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

        if COLLECT_UPDATES:
            eval_callback = UpdatesEvalCallback(
                eval_env,
                log_path=alg_dir_path,
                eval_freq=EVAL_FREQ,
                n_eval_episodes=N_EVAL_EPISODES,
                deterministic=True,
                render=False,
                verbose=0,
            )
        else:
            eval_callback = EvalCallback(
                eval_env,
                log_path=alg_dir_path,
                eval_freq=EVAL_FREQ,
                n_eval_episodes=N_EVAL_EPISODES,
                deterministic=True,
                render=False,
                verbose=0,
            )
        model = GRPO(
            env=vec_env,
            group_rollout_buffer_class=DeepSeekProcessGroupBuffer,
            verbose=args.verbose,
            **model_kwargs,
        )

        print(f"Starting DeepSeek Process GRPO training run {run}...")
        start_time = time.time()

        model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=eval_callback)
        model.save(os.path.join(alg_dir_path, f"model_{run}"))
        data = np.load(os.path.join(alg_dir_path, "evaluations.npz"))
        np.savez(exp_utils.get_experiment_data_path(args.exp_id, algorithm_name, run), **data)

        end_time = time.time()
        print(f"DeepSeek Process GRPO training run {run} completed in {(end_time - start_time):.2f} seconds.")

if RLAlgorithm.OUTCOME_GRPO in args.alg:
    # Outcome supervision GRPO
    algorithm_name = RLAlgorithm.OUTCOME_GRPO
    n_existing_runs = exp_utils.get_existing_runs(args.exp_id, algorithm_name)
    alg_dir_path = exp_utils.get_experiment_data_path(args.exp_id, algorithm_name)
    os.makedirs(alg_dir_path, exist_ok=True)

    # These specified model kwargs will only be used if no algorithm-specific config has been stored yet for the experiment
    model_kwargs = {
        "policy": "GroupPolicy",
        "group_size": 16,
        "n_epochs": 10,
    }
    model_kwargs = exp_utils.get_algorithm_config(args.exp_id, algorithm_name, model_kwargs)

    for run in range(n_existing_runs, args.n_runs + n_existing_runs):
        vec_env = make_vec_env(ENV_NAME, n_envs=1)
        eval_env = make_vec_env(ENV_NAME, n_envs=1)
        if NORMALIZE_ENV:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

        if COLLECT_UPDATES:
            eval_callback = UpdatesEvalCallback(
                eval_env,
                log_path=alg_dir_path,
                eval_freq=EVAL_FREQ,
                n_eval_episodes=N_EVAL_EPISODES,
                deterministic=True,
                render=False,
                verbose=0,
            )
        else:
            eval_callback = EvalCallback(
                eval_env,
                log_path=alg_dir_path,
                eval_freq=EVAL_FREQ,
                n_eval_episodes=N_EVAL_EPISODES,
                deterministic=True,
                render=False,
                verbose=0,
            )
        model = GRPO(
            env=vec_env,
            supervision_type=SupervisionType.OUTCOME,
            group_rollout_buffer_class=DeepSeekOutcomeGroupBuffer,
            verbose=args.verbose,
            **model_kwargs,
        )

        print(f"Starting Outcome GRPO training run {run}...")
        start_time = time.time()

        model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=eval_callback)
        model.save(os.path.join(alg_dir_path, f"model_{run}"))
        data = np.load(os.path.join(alg_dir_path, "evaluations.npz"))
        np.savez(exp_utils.get_experiment_data_path(args.exp_id, algorithm_name, run), **data)

        end_time = time.time()
        print(f"Outcome GRPO training run {run} completed in {(end_time - start_time):.2f} seconds.")

# --- PPO ---
if RLAlgorithm.PPO in args.alg:
    # Standard PPO
    algorithm_name = RLAlgorithm.PPO
    n_existing_runs = exp_utils.get_existing_runs(args.exp_id, algorithm_name)
    alg_dir_path = exp_utils.get_experiment_data_path(args.exp_id, algorithm_name)
    os.makedirs(alg_dir_path, exist_ok=True)

    # These specified model kwargs will only be used if no algorithm-specific config has been stored yet for the experiment
    model_kwargs = {
        "policy": "MlpPolicy",
        "n_epochs": 10,
    }
    model_kwargs = exp_utils.get_algorithm_config(args.exp_id, algorithm_name, model_kwargs)

    for run in range(n_existing_runs, args.n_runs + n_existing_runs):
        vec_env = make_vec_env(ENV_NAME, n_envs=1)
        eval_env = make_vec_env(ENV_NAME, n_envs=1)
        if NORMALIZE_ENV:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

        if COLLECT_UPDATES:
            eval_callback = UpdatesEvalCallback(
                eval_env,
                log_path=alg_dir_path,
                eval_freq=EVAL_FREQ,
                n_eval_episodes=N_EVAL_EPISODES,
                deterministic=True,
                render=False,
                verbose=0,
            )
        else:
            eval_callback = EvalCallback(
                eval_env,
                log_path=alg_dir_path,
                eval_freq=EVAL_FREQ,
                n_eval_episodes=N_EVAL_EPISODES,
                deterministic=True,
                render=False,
                verbose=0,
            )
        model = PPO(
            env=vec_env,
            verbose=args.verbose,
            **model_kwargs,
        )

        print(f"Starting PPO training run {run}...")
        start_time = time.time()

        model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=eval_callback)
        model.save(os.path.join(alg_dir_path, f"model_{run}"))
        data = np.load(os.path.join(alg_dir_path, "evaluations.npz"))
        np.savez(exp_utils.get_experiment_data_path(args.exp_id, algorithm_name, run), **data)

        end_time = time.time()
        print(f"PPO training run {run} completed in {(end_time - start_time):.2f} seconds.")
