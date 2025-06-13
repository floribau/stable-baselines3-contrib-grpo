"""Main module to run GRPO on CartPole-v1 environment."""
import time
import os
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

ENV_NAME = "LunarLander-v3"
N_TRAINING_TIMESTEPS = 100_000

N_EVAL_EPISODES = 4
EVAL_FREQ = 5000
MODELS_PATH = f"./eval/models/{ENV_NAME}/"
EVAL_LOGS_PATH = f"./eval/eval_logs/{ENV_NAME}/"
EVAL_PLOTS_PATH = f"./eval/eval_plots/{ENV_NAME}/"

PLOT_EVAL_RESULTS = True  # Set to True to plot evaluation results after training
EVAL_PLOT_DISPLAY_STEPS = 100_000  # should be <= N_TRAINING_TIMESTEPS
EVAL_PLOT_OPAQUENESS_ALPHA = 0.2
SAVE_EVAL_PLOT = True

# --- Training options ---
# --- RLOO ---
TRAIN_OUTCOME_RLOO = False
TRAIN_OUTCOME_RLOO_NO_GRAD_CLIPPING = False
TRAIN_OUTCOME_RLOO_WITH_KL = False
# --- GRPO ---
TRAIN_PROCESS_GRPO_NO_CLIPPING_NO_KL = False
TRAIN_OUTCOME_GRPO_NO_CLIPPING_NO_KL = False
TRAIN_PROCESS_GRPO_NO_CLIPPING = False
TRAIN_OUTCOME_GRPO_NO_CLIPPING = False
TRAIN_PROCESS_GRPO_NO_KL = False
TRAIN_OUTCOME_GRPO_NO_KL = False
TRAIN_PROCESS_GRPO = False
TRAIN_OUTCOME_GRPO = False
# --- PPO ---
TRAIN_PPO = True

# --- Plotting options ---
# --- RLOO ---
PLOT_OUTCOME_RLOO = True
PLOT_OUTCOME_RLOO_NO_GRAD_CLIPPING = False
PLOT_OUTCOME_RLOO_WITH_KL = False
# --- GRPO ---
PLOT_PROCESS_GRPO_NO_CLIPPING_NO_KL = False
PLOT_OUTCOME_GRPO_NO_CLIPPING_NO_KL = False
PLOT_PROCESS_GRPO_NO_CLIPPING = False
PLOT_OUTCOME_GRPO_NO_CLIPPING = False
PLOT_PROCESS_GRPO_NO_KL = False
PLOT_OUTCOME_GRPO_NO_KL = False
PLOT_PROCESS_GRPO = True
PLOT_OUTCOME_GRPO = True
# --- PPO ---
PLOT_PPO = True

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- Training and Evaluation ---
# --- RLOO ---
if TRAIN_OUTCOME_RLOO:
    # Vanilla RLOO with outcome supervision
    outcome_rloo_vec_env = make_vec_env(ENV_NAME, n_envs=1)
    outcome_rloo_eval_env = make_vec_env(ENV_NAME, n_envs=1)

    outcome_rloo_eval_callback = EvalCallback(
        outcome_rloo_eval_env,
        log_path=EVAL_LOGS_PATH + "outcome_rloo/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    outcome_rloo_model = RLOO(
        "GroupPolicy",
        outcome_rloo_vec_env,
        verbose=1,
        group_size=16,
        kl_beta=0,  # no KL penalty in standard RLOO
    )

    print("Starting Outcome RLOO training...")
    start_time = time.time()

    # pr = cProfile.Profile()
    # print("Starting Outcome RLOO profiling...")
    # pr.enable()  # Start profiling

    outcome_rloo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=outcome_rloo_eval_callback)
    # rloo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS)
    outcome_rloo_model.save(f"{MODELS_PATH}outcome_rloo_model")

    # pr.disable()
    # print("Outcome RLOO profiling completed. Saving profiling data...")
    # pr.dump_stats(f"./eval/profiling_outputs/outcome_rloo_profile_{timestamp}.prof")

    end_time = time.time()
    print(f"Outcome RLOO training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_OUTCOME_RLOO_NO_GRAD_CLIPPING:
    # Pure RLOO
    outcome_rloo_no_grad_clipping_vec_env = make_vec_env(ENV_NAME, n_envs=1)
    outcome_rloo_no_grad_clipping_eval_env = make_vec_env(ENV_NAME, n_envs=1)

    outcome_rloo_no_grad_clipping_eval_callback = EvalCallback(
        outcome_rloo_no_grad_clipping_eval_env,
        log_path=EVAL_LOGS_PATH + "outcome_rloo_no_grad_clipping/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    outcome_rloo_no_grad_clipping_model = RLOO(
        "GroupPolicy",
        outcome_rloo_no_grad_clipping_vec_env,
        verbose=1,
        group_size=16,
        kl_beta=0,  # no KL penalty in standard RLOO
        max_grad_norm=None,  # no grad clipping
    )

    print("Starting Outcome RLOO without grad clipping training...")
    start_time = time.time()

    # pr = cProfile.Profile()
    # print("Starting Outcome RLOO without grad clipping profiling...")
    # pr.enable()  # Start profiling

    outcome_rloo_no_grad_clipping_model.learn(
        total_timesteps=N_TRAINING_TIMESTEPS, callback=outcome_rloo_no_grad_clipping_eval_callback
    )
    # rloo_no_grad_clipping_model.learn(total_timesteps=N_TRAINING_TIMESTEPS)
    outcome_rloo_no_grad_clipping_model.save(f"{MODELS_PATH}outcome_rloo_no_grad_clipping_model")

    # pr.disable()
    # print("Outcome RLOO without grad clipping profiling completed. Saving profiling data...")
    # pr.dump_stats(f"./eval/profiling_outputs/rloo_no_grad_clipping_profile_{timestamp}.prof")

    end_time = time.time()
    print(f"Outcome RLOO without grad clipping training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_OUTCOME_RLOO_WITH_KL:
    # Outcome supervision RLOO with KL penalty
    outcome_rloo_kl_vec_env = make_vec_env(ENV_NAME, n_envs=1)
    outcome_rloo_kl_eval_env = make_vec_env(ENV_NAME, n_envs=1)

    outcome_rloo_kl_eval_callback = EvalCallback(
        outcome_rloo_kl_eval_env,
        log_path=EVAL_LOGS_PATH + "outcome_rloo_kl/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    outcome_rloo_kl_model = RLOO(
        "GroupPolicy",
        outcome_rloo_kl_vec_env,
        kl_ref_iterations=10,
        verbose=1,
        group_size=16,
    )

    print("Starting Outcome RLOO with KL training...")
    start_time = time.time()

    outcome_rloo_kl_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=outcome_rloo_kl_eval_callback)
    outcome_rloo_kl_model.save(f"{MODELS_PATH}outcome_rloo_kl_model")

    end_time = time.time()
    print(f"Outcome RLOO with KL training completed in {(end_time - start_time):.2f} seconds.")

# --- GRPO ---
if TRAIN_PROCESS_GRPO_NO_CLIPPING_NO_KL:
    # Process supervision RLOO with Importance Sampling
    process_grpo_no_clipping_no_kl_vec_env = make_vec_env(ENV_NAME, n_envs=1)
    process_grpo_no_clipping_no_kl_eval_env = make_vec_env(ENV_NAME, n_envs=1)

    process_grpo_no_clipping_no_kl_eval_callback = EvalCallback(
        process_grpo_no_clipping_no_kl_eval_env,
        log_path=EVAL_LOGS_PATH + "process_grpo_no_clipping_no_kl/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    process_grpo_no_clipping_no_kl_model = GRPO(
        "GroupPolicy",
        process_grpo_no_clipping_no_kl_vec_env,
        verbose=1,
        group_size=16,
        n_epochs=10,
        kl_beta=0,  # Set beta to zero, no KL penalty
        clip_range=float("inf"),  # Set clipping factor to infinity for RLOO with IS
    )

    print("Starting Process GRPO without clipping and KL training...")
    start_time = time.time()

    process_grpo_no_clipping_no_kl_model.learn(
        total_timesteps=N_TRAINING_TIMESTEPS, callback=process_grpo_no_clipping_no_kl_eval_callback
    )
    process_grpo_no_clipping_no_kl_model.save(f"{MODELS_PATH}process_grpo_no_clipping_no_kl_model")

    end_time = time.time()
    print(f"Process RPO without clipping and KL training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_OUTCOME_GRPO_NO_CLIPPING_NO_KL:
    # Outcome supervision RLOO with Importance Sampling
    outcome_grpo_no_clipping_no_kl_vec_env = make_vec_env(ENV_NAME, n_envs=1)
    outcome_grpo_no_clipping_no_kl_eval_env = make_vec_env(ENV_NAME, n_envs=1)

    outcome_grpo_no_clipping_no_kl_eval_callback = EvalCallback(
        outcome_grpo_no_clipping_no_kl_eval_env,
        log_path=EVAL_LOGS_PATH + "outcome_grpo_no_clipping_no_kl/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    outcome_grpo_no_clipping_no_kl_model = GRPO(
        "GroupPolicy",
        outcome_grpo_no_clipping_no_kl_vec_env,
        verbose=1,
        group_size=16,
        n_epochs=10,
        kl_beta=0,  # Set beta to zero, no KL penalty
        clip_range=float("inf"),  # Set clipping factor to infinity for RLOO with IS
    )

    print("Starting Outcome GRPO without clipping and KL training...")
    start_time = time.time()

    outcome_grpo_no_clipping_no_kl_model.learn(
        total_timesteps=N_TRAINING_TIMESTEPS, callback=outcome_grpo_no_clipping_no_kl_eval_callback
    )
    outcome_grpo_no_clipping_no_kl_model.save(f"{MODELS_PATH}outcome_grpo_no_clipping_no_kl_model")

    end_time = time.time()
    print(f"Outcome GRPO without clipping and KL training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_PROCESS_GRPO_NO_CLIPPING:
    # Process supervision RLOO with KL penalty and Importance Sampling
    process_grpo_no_clipping_vec_env = make_vec_env(ENV_NAME, n_envs=1)
    process_grpo_no_clipping_eval_env = make_vec_env(ENV_NAME, n_envs=1)

    process_grpo_no_clipping_eval_callback = EvalCallback(
        process_grpo_no_clipping_eval_env,
        log_path=EVAL_LOGS_PATH + "process_grpo_no_clipping/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    process_grpo_no_clipping_model = GRPO(
        "GroupPolicy",
        process_grpo_no_clipping_vec_env,
        verbose=1,
        group_size=16,
        n_epochs=10,
        clip_range=float("inf"),  # Set clipping factor to infinity for RLOO with KL penalty and IS
    )

    print("Starting Process GRPO without clipping training...")
    start_time = time.time()

    process_grpo_no_clipping_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=process_grpo_no_clipping_eval_callback)
    process_grpo_no_clipping_model.save(f"{MODELS_PATH}process_grpo_no_clipping_model")

    end_time = time.time()
    print(f"Process without clipping training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_OUTCOME_GRPO_NO_CLIPPING:
    # Outcome supvervision RLOO with KL penalty and Importance Sampling
    outcome_grpo_no_clipping_vec_env = make_vec_env(ENV_NAME, n_envs=1)
    outcome_grpo_no_clipping_eval_env = make_vec_env(ENV_NAME, n_envs=1)

    outcome_grpo_no_clipping_eval_callback = EvalCallback(
        outcome_grpo_no_clipping_eval_env,
        log_path=EVAL_LOGS_PATH + "outcome_grpo_no_clipping/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    outcome_grpo_no_clipping_model = GRPO(
        "GroupPolicy",
        outcome_grpo_no_clipping_vec_env,
        verbose=1,
        group_size=16,
        n_epochs=10,
        group_rollout_buffer_class=OutcomeGroupBuffer,
        clip_range=float("inf"),  # Set clipping factor to infinity for RLOO with KL penalty and IS
    )

    print("Starting Outcome GRPO without clipping training...")
    start_time = time.time()

    outcome_grpo_no_clipping_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=outcome_grpo_no_clipping_eval_callback)
    outcome_grpo_no_clipping_model.save(f"{MODELS_PATH}outcome_grpo_no_clipping_model")

    end_time = time.time()
    print(f"Outcome without clipping training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_PROCESS_GRPO_NO_KL:
    # Process supervision RLOO with clipping
    process_grpo_no_kl_vec_env = make_vec_env(ENV_NAME, n_envs=1)
    process_grpo_no_kl_eval_env = make_vec_env(ENV_NAME, n_envs=1)

    process_grpo_no_kl_eval_callback = EvalCallback(
        process_grpo_no_kl_eval_env,
        log_path=EVAL_LOGS_PATH + "process_grpo_no_kl/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    process_grpo_no_kl_model = GRPO(
        "GroupPolicy",
        process_grpo_no_kl_vec_env,
        verbose=1,
        group_size=16,
        n_epochs=10,
        kl_beta=0,  # No KL penalty for clipping
    )

    print("Starting Process GRPO without KL training...")
    start_time = time.time()

    process_grpo_no_kl_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=process_grpo_no_kl_eval_callback)
    process_grpo_no_kl_model.save(f"{MODELS_PATH}process_grpo_no_kl_model")

    end_time = time.time()
    print(f"Process GRPO without KL training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_OUTCOME_GRPO_NO_KL:
    # Outcome supervision RLOO with clipping
    outcome_grpo_no_kl_vec_env = make_vec_env(ENV_NAME, n_envs=1)
    outcome_grpo_no_kl_eval_env = make_vec_env(ENV_NAME, n_envs=1)

    outcome_grpo_no_kl_eval_callback = EvalCallback(
        outcome_grpo_no_kl_eval_env,
        log_path=EVAL_LOGS_PATH + "outcome_grpo_no_kl/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    outcome_grpo_no_kl_model = GRPO(
        "GroupPolicy",
        outcome_grpo_no_kl_vec_env,
        verbose=1,
        group_size=16,
        n_epochs=10,
        kl_beta=0,
        group_rollout_buffer_class=OutcomeGroupBuffer,
    )

    print("Starting Outcome GRPO without KL training...")
    start_time = time.time()

    outcome_grpo_no_kl_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=outcome_grpo_no_kl_eval_callback)
    outcome_grpo_no_kl_model.save(f"{MODELS_PATH}outcome_grpo_no_kl_model")

    end_time = time.time()
    print(f"Outcome GRPO without KL training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_PROCESS_GRPO:
    # Process supevision GRPO
    process_grpo_vec_env = make_vec_env(ENV_NAME, n_envs=1)
    process_grpo_eval_env = make_vec_env(ENV_NAME, n_envs=1)

    process_grpo_eval_callback = EvalCallback(
        process_grpo_eval_env,
        log_path=EVAL_LOGS_PATH + "process_grpo/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    process_grpo_model = GRPO(
        "GroupPolicy",
        process_grpo_vec_env,
        verbose=1,
        group_size=16,
        n_epochs=10,
    )

    print("Starting Process GRPO training...")
    start_time = time.time()

    # pr = cProfile.Profile()
    # print("Starting Process GRPO profiling...")
    # pr.enable()  # Start profiling

    process_grpo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=process_grpo_eval_callback)
    # process_grpo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS)
    process_grpo_model.save(f"{MODELS_PATH}process_grpo_model")

    # pr.disable()
    # print("Process GRPO profiling completed. Saving profiling data...")
    # pr.dump_stats(f"./eval/profiling_outputs/process_grpo_profile_{timestamp}.prof")

    end_time = time.time()
    print(f"Process GRPO training completed in {(end_time - start_time):.2f} seconds.")

if TRAIN_OUTCOME_GRPO:
    # Outcome supervision GRPO
    outcome_grpo_vec_env = make_vec_env(ENV_NAME, n_envs=1)
    outcome_grpo_eval_env = make_vec_env(ENV_NAME, n_envs=1)

    outcome_grpo_eval_callback = EvalCallback(
        outcome_grpo_eval_env,
        log_path=EVAL_LOGS_PATH + "outcome_grpo/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    outcome_grpo_model = GRPO(
        "GroupPolicy",
        outcome_grpo_vec_env,
        verbose=1,
        group_size=16,
        n_epochs=10,
        group_rollout_buffer_class=OutcomeGroupBuffer,
    )

    print("Starting Outcome GRPO training...")
    start_time = time.time()

    # pr = cProfile.Profile()
    # print("Starting Outcome GRPO profiling...")
    # pr.enable()  # Start profiling

    outcome_grpo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=outcome_grpo_eval_callback)
    # outcome_grpo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS)
    outcome_grpo_model.save(f"{MODELS_PATH}outcome_grpo_model")

    # pr.disable()
    # print("Outcome GRPO profiling completed. Saving profiling data...")
    # pr.dump_stats(f"./eval/profiling_outputs/outcome_grpo_profile_{timestamp}.prof")

    end_time = time.time()
    print(f"Outcome GRPO training completed in {(end_time - start_time):.2f} seconds.")

# --- PPO ---
if TRAIN_PPO:
    # Standard PPO
    ppo_vec_env = make_vec_env(ENV_NAME, n_envs=1)
    ppo_eval_env = make_vec_env(ENV_NAME, n_envs=1)

    ppo_eval_callback = EvalCallback(
        ppo_eval_env,
        log_path=EVAL_LOGS_PATH + "ppo/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=0,
    )
    ppo_model = PPO(
        "MlpPolicy",
        ppo_vec_env,
        verbose=1,
        n_epochs=10,
    )

    print("Starting PPO training...")
    start_time = time.time()

    # pr = cProfile.Profile()
    # print("Starting PPO profiling...")
    # pr.enable()  # Start profiling

    ppo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS, callback=ppo_eval_callback)
    # ppo_model.learn(total_timesteps=N_TRAINING_TIMESTEPS)
    ppo_model.save(f"{MODELS_PATH}ppo_model")

    # NOTE only for visual inspection
    obs = ppo_vec_env.reset()
    for _ in range(1000):
        action, _states = ppo_model.predict(obs)
        obs, rewards, dones, info = ppo_vec_env.step(action)
        ppo_vec_env.render("human")

    # pr.disable()
    # print("PPO profiling completed. Saving profiling data...")
    # pr.dump_stats(f"./eval/profiling_outputs/ppo_profile_{timestamp}.prof")

    end_time = time.time()
    print(f"PPO training completed in {(end_time - start_time):.2f} seconds.")

# --- Plotting Evaluation Results ---
if PLOT_EVAL_RESULTS:
    # --- RLOO ---
    if PLOT_OUTCOME_RLOO:
        data = np.load(EVAL_LOGS_PATH + "outcome_rloo/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Outcome RLOO Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_OUTCOME_RLOO_NO_GRAD_CLIPPING:
        data = np.load(EVAL_LOGS_PATH + "outcome_rloo_no_grad_clipping/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Outcome RLOO no grad clipping Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_OUTCOME_RLOO_WITH_KL:
        data = np.load(EVAL_LOGS_PATH + "outcome_rloo_kl/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Outcome RLOO with KL Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    # --- GRPO ---
    if PLOT_PROCESS_GRPO_NO_CLIPPING_NO_KL:
        data = np.load(EVAL_LOGS_PATH + "process_grpo_no_clipping_no_kl/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Process GRPO no clipping no KL Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_OUTCOME_GRPO_NO_CLIPPING_NO_KL:
        data = np.load(EVAL_LOGS_PATH + "outcome_grpo_no_clipping_no_kl/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Outcome GRPO no clipping no KL Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_PROCESS_GRPO_NO_CLIPPING:
        data = np.load(EVAL_LOGS_PATH + "process_grpo_no_clipping/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Process GRPO no clipping Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_OUTCOME_GRPO_NO_CLIPPING:
        data = np.load(EVAL_LOGS_PATH + "outcome_grpo_no_clipping/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Outcome GRPO no clipping Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_PROCESS_GRPO_NO_KL:
        data = np.load(EVAL_LOGS_PATH + "process_grpo_no_kl/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Process GRPO no KL Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_OUTCOME_GRPO_NO_KL:
        data = np.load(EVAL_LOGS_PATH + "outcome_grpo_no_kl/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Outcome GRPO no KL Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_PROCESS_GRPO:
        data = np.load(EVAL_LOGS_PATH + "process_grpo/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Process GRPO Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    if PLOT_OUTCOME_GRPO:
        data = np.load(EVAL_LOGS_PATH + "outcome_grpo/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="Outcome GRPO Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    # --- PPO ---
    if PLOT_PPO:
        data = np.load(EVAL_LOGS_PATH + "ppo/evaluations.npz")
        timesteps = data["timesteps"]
        timesteps = timesteps[:np.searchsorted(timesteps, EVAL_PLOT_DISPLAY_STEPS, side='right')]
        results = data["results"]
        results = results[:len(timesteps)]

        mean_rewards = results.mean(axis=1)
        std_rewards = results.std(axis=1)

        plt.plot(timesteps, mean_rewards, label="PPO Reward")
        plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation Reward")
    plt.title("Evaluation Performance Over Time")
    plt.legend(loc="best", framealpha=0.3)
    plt.grid()
    plt.tight_layout()
    if SAVE_EVAL_PLOT:
        os.makedirs(EVAL_PLOTS_PATH, exist_ok=True)
        plt.savefig(f"{EVAL_PLOTS_PATH}eval_performance_{timestamp}.png")
    plt.show()
