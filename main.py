"""Main module to run GRPO on CartPole-v1 environment."""
import time
from datetime import datetime
import warnings
import cProfile

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.ppo import PPO
from sb3_contrib.grpo.grpo import GRPO
from sb3_contrib.rloo.rloo import RLOO
from sb3_contrib.grpo.buffers import OutcomeGroupBuffer

warnings.filterwarnings("error", category=RuntimeWarning)  # DEBUG line for temporarily converting warnings to errors

N_TRAINING_TIMESTEPS = 300_000

N_EVAL_EPISODES = 5
EVAL_FREQ = 1000
EVAL_PATH = "./eval/eval_logs/"

PLOT_EVAL_RESULTS = True  # Set to True to plot evaluation results after training
EVAL_PLOT_DISPLAY_STEPS = 300_000  # should be <= N_TRAINING_TIMESTEPS
EVAL_PLOT_OPAQUENESS_ALPHA = 0.2
SAVE_EVAL_PLOT = True

TRAIN_PROCESS_RLOO = False
TRAIN_OUTCOME_RLOO = False
TRAIN_PROCESS_RLOO_WITH_KL = False
TRAIN_OUTCOME_RLOO_WITH_KL = False
TRAIN_PROCESS_RLOO_WITH_KL_IS = False
TRAIN_OUTCOME_RLOO_WITH_KL_IS = False
TRAIN_PROCESS_RLOO_WITH_IS = False
TRAIN_OUTCOME_RLOO_WITH_IS = False
TRAIN_PROCESS_RLOO_WITH_CLIPPING = False
TRAIN_OUTCOME_RLOO_WITH_CLIPPING = False
TRAIN_PROCESS_GRPO = False
TRAIN_OUTCOME_GRPO = False
TRAIN_PPO = False
TRAIN_PURE_RLOO = False

PLOT_PROCESS_RLOO = False
PLOT_OUTCOME_RLOO = False
PLOT_PROCESS_RLOO_WITH_KL = False
PLOT_OUTCOME_RLOO_WITH_KL = False
PLOT_PROCESS_RLOO_WITH_IS = False
PLOT_OUTCOME_RLOO_WITH_IS = False
PLOT_PROCESS_RLOO_WITH_KL_IS = False
PLOT_OUTCOME_RLOO_WITH_KL_IS = False
PLOT_PROCESS_RLOO_WITH_CLIPPING = False
PLOT_OUTCOME_RLOO_WITH_CLIPPING = False
PLOT_PROCESS_GRPO = True
PLOT_OUTCOME_GRPO = True
PLOT_PPO = True
PLOT_PURE_RLOO = True

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- Training and Evaluation ---
if TRAIN_PROCESS_RLOO:
    # Vanilla RLOO with process supervision
    process_rloo_vec_env = make_vec_env("CartPole-v1", n_envs=1)
    process_rloo_eval_env = make_vec_env("CartPole-v1", n_envs=1)

    process_rloo_eval_callback = EvalCallback(
        eval_env=process_rloo_eval_env,
        log_path=EVAL_PATH + "process_rloo/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )

    process_rloo_model = GRPO(
        "GroupPolicy",
        process_rloo_vec_env,
        verbose=0,
        group_size=16,
        n_epochs=10,
        kl_beta=0.0,  # Set beta to zero for vanilla RLOO, no KL penalty
        clip_range=float("inf"),  # Set clipping factor to infinity for vanilla RLOO
        use_importance_sampling=False,  # Set IS flag to False for vanilla RLOO
    )
    print("Starting Process RLOO training...")
    start_time = time.time()
    process_rloo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=process_rloo_eval_callback)
    end_time = time.time()
    print(f"Process RLOO training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_OUTCOME_RLOO:
    # Vanilla RLOO with outcome supervision
    outcome_rloo_vec_env = make_vec_env("CartPole-v1", n_envs=1)
    outcome_rloo_eval_env = make_vec_env("CartPole-v1", n_envs=1)

    outcome_rloo_eval_callback = EvalCallback(
        outcome_rloo_eval_env,
        log_path=EVAL_PATH + "outcome_rloo/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )

    outcome_rloo_model = GRPO(
        "GroupPolicy",
        outcome_rloo_vec_env,
        verbose=0,
        group_size=16,
        n_epochs=10,
        group_rollout_buffer_class=OutcomeGroupBuffer,
        kl_beta=0.0,  # Set beta to zero for vanilla RLOO, no KL penalty
        clip_range=float("inf"),  # Set clipping factor to infinity for vanilla RLOO
        use_importance_sampling=False,  # Set IS flag to False for vanilla RLOO
    )
    print("Starting Outcome RLOO training...")
    start_time = time.time()
    outcome_rloo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=outcome_rloo_eval_callback)
    end_time = time.time()
    print(f"Outcome RLOO training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_PROCESS_RLOO_WITH_KL:
    # Process supervision RLOO with KL penalty
    process_rloo_kl_vec_env = make_vec_env("CartPole-v1", n_envs=1)
    process_rloo_kl_eval_env = make_vec_env("CartPole-v1", n_envs=1)

    process_rloo_kl_eval_callback = EvalCallback(
        process_rloo_kl_eval_env,
        log_path=EVAL_PATH + "process_rloo_kl/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )

    process_rloo_kl_model = GRPO(
        "GroupPolicy",
        process_rloo_kl_vec_env,
        verbose=0,
        group_size=16,
        n_epochs=10,
        clip_range=float("inf"),  # Set clipping factor to infinity for RLOO with KL penalty
        use_importance_sampling=False,  # Set IS flag to False for RLOO with KL penalty
    )
    print("Starting Process RLOO with KL training...")
    start_time = time.time()
    process_rloo_kl_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=process_rloo_kl_eval_callback)
    end_time = time.time()
    print(f"Process RLOO with KL training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_OUTCOME_RLOO_WITH_KL:
    # Outcome supervision RLOO with KL penalty
    outcome_rloo_kl_vec_env = make_vec_env("CartPole-v1", n_envs=1)
    outcome_rloo_kl_eval_env = make_vec_env("CartPole-v1", n_envs=1)

    outcome_rloo_kl_eval_callback = EvalCallback(
        outcome_rloo_kl_eval_env,
        log_path=EVAL_PATH + "outcome_rloo_kl/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )

    outcome_rloo_kl_model = GRPO(
        "GroupPolicy",
        outcome_rloo_kl_vec_env,
        verbose=0,
        group_size=16,
        n_epochs=10,
        group_rollout_buffer_class=OutcomeGroupBuffer,
        clip_range=float("inf"),  # Set clipping factor to infinity for RLOO with KL penalty
        use_importance_sampling=False,  # Set IS flag to False for RLOO with KL penalty
    )
    print("Starting Outcome RLOO with KL training...")
    start_time = time.time()
    outcome_rloo_kl_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=outcome_rloo_kl_eval_callback)
    end_time = time.time()
    print(f"Outcome RLOO with KL training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_PROCESS_RLOO_WITH_IS:
    # Process supervision RLOO with Importance Sampling
    process_rloo_is_vec_env = make_vec_env("CartPole-v1", n_envs=1)
    process_rloo_is_eval_env = make_vec_env("CartPole-v1", n_envs=1)

    process_rloo_is_eval_callback = EvalCallback(
        process_rloo_is_eval_env,
        log_path=EVAL_PATH + "process_rloo_is/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )

    process_rloo_is_model = GRPO(
        "GroupPolicy",
        process_rloo_is_vec_env,
        verbose=0,
        group_size=16,
        n_epochs=10,
        kl_beta=0.0,  # Set beta to zero, no KL penalty
        clip_range=float("inf"),  # Set clipping factor to infinity for RLOO with IS
    )
    print("Starting Process RLOO with IS training...")
    start_time = time.time()
    process_rloo_is_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=process_rloo_is_eval_callback)
    end_time = time.time()
    print(f"Process RLOO with IS training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_OUTCOME_RLOO_WITH_IS:
    # Outcome supervision RLOO with Importance Sampling
    outcome_rloo_is_vec_env = make_vec_env("CartPole-v1", n_envs=1)
    outcome_rloo_is_eval_env = make_vec_env("CartPole-v1", n_envs=1)

    outcome_rloo_is_eval_callback = EvalCallback(
        outcome_rloo_is_eval_env,
        log_path=EVAL_PATH + "outcome_rloo_is/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )

    outcome_rloo_is_model = GRPO(
        "GroupPolicy",
        outcome_rloo_is_vec_env,
        verbose=0,
        group_size=16,
        n_epochs=10,
        kl_beta=0.0,  # Set beta to zero, no KL penalty
        clip_range=float("inf"),  # Set clipping factor to infinity for RLOO with IS
    )
    print("Starting Outcome RLOO with IS training...")
    start_time = time.time()
    outcome_rloo_is_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=outcome_rloo_is_eval_callback)
    end_time = time.time()
    print(f"Outcome RLOO with IS training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_PROCESS_RLOO_WITH_KL_IS:
    # Process supervision RLOO with KL penalty and Importance Sampling
    process_rloo_kl_is_vec_env = make_vec_env("CartPole-v1", n_envs=1)
    process_rloo_kl_is_eval_env = make_vec_env("CartPole-v1", n_envs=1)

    process_rloo_kl_is_eval_callback = EvalCallback(
        process_rloo_kl_is_eval_env,
        log_path=EVAL_PATH + "process_rloo_kl_is/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )

    process_rloo_kl_is_model = GRPO(
        "GroupPolicy",
        process_rloo_kl_is_vec_env,
        verbose=0,
        group_size=16,
        n_epochs=10,
        clip_range=float("inf"),  # Set clipping factor to infinity for RLOO with KL penalty and IS
    )
    print("Starting Process RLOO with KL and IS training...")
    start_time = time.time()
    process_rloo_kl_is_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=process_rloo_kl_is_eval_callback)
    end_time = time.time()
    print(f"Process RLOO with KL and IS training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_OUTCOME_RLOO_WITH_KL_IS:
    # Outcome supvervision RLOO with KL penalty and Importance Sampling
    outcome_rloo_kl_is_vec_env = make_vec_env("CartPole-v1", n_envs=1)
    outcome_rloo_kl_is_eval_env = make_vec_env("CartPole-v1", n_envs=1)

    outcome_rloo_kl_is_eval_callback = EvalCallback(
        outcome_rloo_kl_is_eval_env,
        log_path=EVAL_PATH + "outcome_rloo_kl_is/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )

    outcome_rloo_kl_is_model = GRPO(
        "GroupPolicy",
        outcome_rloo_kl_is_vec_env,
        verbose=0,
        group_size=16,
        n_epochs=10,
        group_rollout_buffer_class=OutcomeGroupBuffer,
        clip_range=float("inf"),  # Set clipping factor to infinity for RLOO with KL penalty and IS
    )
    print("Starting Outcome RLOO with KL and IS training...")
    start_time = time.time()
    outcome_rloo_kl_is_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=outcome_rloo_kl_is_eval_callback)
    end_time = time.time()
    print(f"Outcome RLOO with KL and IS training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_PROCESS_RLOO_WITH_CLIPPING:
    # Process supervision RLOO with clipping
    process_rloo_clipping_vec_env = make_vec_env("CartPole-v1", n_envs=1)
    process_rloo_clipping_eval_env = make_vec_env("CartPole-v1", n_envs=1)

    process_rloo_clipping_eval_callback = EvalCallback(
        process_rloo_clipping_eval_env,
        log_path=EVAL_PATH + "process_rloo_clipping/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )

    process_rloo_clipping_model = GRPO(
        "GroupPolicy",
        process_rloo_clipping_vec_env,
        verbose=0,
        group_size=16,
        n_epochs=10,
        kl_beta=0.0,  # No KL penalty for clipping
    )
    print("Starting Process RLOO with Clipping training...")
    start_time = time.time()
    process_rloo_clipping_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=process_rloo_clipping_eval_callback)
    end_time = time.time()
    print(f"Process RLOO with Clipping training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_OUTCOME_RLOO_WITH_CLIPPING:
    # Outcome supervision RLOO with clipping
    outcome_rloo_clipping_vec_env = make_vec_env("CartPole-v1", n_envs=1)
    outcome_rloo_clipping_eval_env = make_vec_env("CartPole-v1", n_envs=1)

    outcome_rloo_clipping_eval_callback = EvalCallback(
        outcome_rloo_clipping_eval_env,
        log_path=EVAL_PATH + "outcome_rloo_clipping/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )

    outcome_rloo_clipping_model = GRPO(
        "GroupPolicy",
        outcome_rloo_clipping_vec_env,
        verbose=0,
        group_size=16,
        n_epochs=10,
        kl_beta=0.0,
        group_rollout_buffer_class=OutcomeGroupBuffer,
    )
    print("Starting Outcome RLOO with Clipping training...")
    start_time = time.time()
    outcome_rloo_clipping_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=outcome_rloo_clipping_eval_callback)
    end_time = time.time()
    print(f"Outcome RLOO with Clipping training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_PROCESS_GRPO:
    # Process supevision GRPO
    process_grpo_vec_env = make_vec_env("CartPole-v1", n_envs=1)
    process_grpo_eval_env = make_vec_env("CartPole-v1", n_envs=1)

    process_grpo_eval_callback = EvalCallback(
        process_grpo_eval_env,
        log_path=EVAL_PATH + "process_grpo/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    grpo_model = GRPO(
        "GroupPolicy",
        process_grpo_vec_env,
        verbose=1,
        group_size=16,
        n_epochs=10,
    )
    print("Starting Process GRPO training...")
    start_time = time.time()

    pr = cProfile.Profile()
    print("Starting Process GRPO profiling...")
    pr.enable()  # Start profiling

    grpo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=process_grpo_eval_callback)
    # grpo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS)

    pr.disable()
    print("Process GRPO profiling completed. Saving profiling data...")
    # pr.dump_stats(f"./eval/profiling_outputs/process_grpo_profile_{timestamp}.prof")

    end_time = time.time()
    print(f"Process GRPO training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_OUTCOME_GRPO:
    # Outcome supervision GRPO
    outcome_grpo_vec_env = make_vec_env("CartPole-v1", n_envs=1)
    outcome_grpo_eval_env = make_vec_env("CartPole-v1", n_envs=1)

    outcome_grpo_eval_callback = EvalCallback(
        outcome_grpo_eval_env,
        log_path=EVAL_PATH + "outcome_grpo/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    grpo_model = GRPO(
        "GroupPolicy",
        outcome_grpo_vec_env,
        verbose=1,
        group_size=16,
        n_epochs=10,
        group_rollout_buffer_class=OutcomeGroupBuffer,
    )

    print("Starting Outcome GRPO training...")
    start_time = time.time()

    pr = cProfile.Profile()
    print("Starting Outcome GRPO profiling...")
    pr.enable()  # Start profiling

    grpo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=outcome_grpo_eval_callback)
    # grpo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS)

    pr.disable()
    print("Outcome GRPO profiling completed. Saving profiling data...")
    # pr.dump_stats(f"./eval/profiling_outputs/outcome_grpo_profile_{timestamp}.prof")

    end_time = time.time()
    print(f"Outcome GRPO training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_PPO:
    # Standard PPO
    ppo_vec_env = make_vec_env("CartPole-v1", n_envs=1)
    ppo_eval_env = make_vec_env("CartPole-v1", n_envs=1)

    ppo_eval_callback = EvalCallback(
        ppo_eval_env,
        log_path=EVAL_PATH + "ppo/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    ppo_model = PPO(
        "MlpPolicy",
        ppo_vec_env,
        verbose=0,
        n_epochs=10,
    )

    print("Starting PPO training...")
    start_time = time.time()

    pr = cProfile.Profile()
    print("Starting PPO profiling...")
    pr.enable()  # Start profiling

    # ppo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=ppo_eval_callback)
    ppo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS)

    pr.disable()
    print("PPO profiling completed. Saving profiling data...")
    # pr.dump_stats(f"./eval/profiling_outputs/ppo_profile_{timestamp}.prof")

    end_time = time.time()
    print(f"PPO training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_PURE_RLOO:
    # Pure RLOO
    vec_env = make_vec_env("CartPole-v1", n_envs=1)
    eval_env = make_vec_env("CartPole-v1", n_envs=1)

    outcome_grpo_eval_callback = EvalCallback(
        eval_env,
        log_path=EVAL_PATH + "pure_rloo_no_clipping/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    grpo_model = RLOO(
        "GroupPolicy",
        vec_env,
        verbose=1,
        group_size=16,
        max_grad_norm=None,
    )

    print("Starting Pure RLOO training...")
    start_time = time.time()

    pr = cProfile.Profile()
    print("Starting Pure RLOO profiling...")
    pr.enable()  # Start profiling

    grpo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=outcome_grpo_eval_callback)
    # grpo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS)

    pr.disable()
    # print("Pure RLOO profiling completed. Saving profiling data...")
    # pr.dump_stats(f"./eval/profiling_outputs/outcome_grpo_profile_{timestamp}.prof")

    end_time = time.time()
    print(f"Pure RLOO training completed in {(end_time - start_time):.2f} seconds.")

# --- Plotting Evaluation Results ---
if PLOT_EVAL_RESULTS:
    if PLOT_PROCESS_RLOO:
        data = np.load(EVAL_PATH + "process_rloo/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Process RLOO Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_OUTCOME_RLOO:
        data = np.load(EVAL_PATH + "outcome_rloo/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Outcome RLOO Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_PROCESS_RLOO_WITH_KL:
        data = np.load(EVAL_PATH + "process_rloo_kl/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Process RLOO with KL Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_OUTCOME_RLOO_WITH_KL:
        data = np.load(EVAL_PATH + "outcome_rloo_kl/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Outcome RLOO with KL Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_PROCESS_RLOO_WITH_IS:
        data = np.load(EVAL_PATH + "process_rloo_is/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Process RLOO with IS Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_OUTCOME_RLOO_WITH_IS:
        data = np.load(EVAL_PATH + "outcome_rloo_is/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Outcome RLOO with IS Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_PROCESS_RLOO_WITH_KL_IS:
        data = np.load(EVAL_PATH + "process_rloo_kl_is/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Process RLOO with KL and IS Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_OUTCOME_RLOO_WITH_KL_IS:
        data = np.load(EVAL_PATH + "outcome_rloo_kl_is/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Outcome RLOO with KL and IS Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_PROCESS_RLOO_WITH_CLIPPING:
        data = np.load(EVAL_PATH + "process_rloo_clipping/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Process RLOO with Clipping Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_OUTCOME_RLOO_WITH_CLIPPING:
        data = np.load(EVAL_PATH + "outcome_rloo_clipping/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Outcome RLOO with Clipping Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_PROCESS_GRPO:
        data = np.load(EVAL_PATH + "process_grpo/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Process GRPO Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_OUTCOME_GRPO:
        data = np.load(EVAL_PATH + "outcome_grpo/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Outcome GRPO Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_PPO:
        data = np.load(EVAL_PATH + "ppo/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="PPO Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_PURE_RLOO:
        data = np.load(EVAL_PATH + "pure_rloo/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Pure RLOO Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

        data = np.load(EVAL_PATH + "pure_rloo_no_clipping/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Pure RLOO No Clipping Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation Reward")
    plt.title("Evaluation Performance Over Time")
    plt.legend(loc="best", framealpha=0.3)
    plt.grid()
    plt.tight_layout()
    if SAVE_EVAL_PLOT:
        plt.savefig(f"./eval/eval_images/eval_performance_{timestamp}.png")
    plt.show()
