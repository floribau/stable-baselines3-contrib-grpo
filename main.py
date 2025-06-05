"""Main module to run GRPO on CartPole-v1 environment."""
import time
from datetime import datetime
import warnings

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.ppo import PPO
from sb3_contrib.grpo.grpo import GRPO
from sb3_contrib.grpo.buffers import OutcomeGroupBuffer

warnings.filterwarnings("error", category=RuntimeWarning)  # DEBUG line for temporarily converting warnings to errors

TRAINING_TIMESTEPS = 200_000
N_EVAL_EPISODES = 5
EVAL_FREQ = 1000
EVAL_PATH = "./eval_logs/"
SAVE_EVAL_PLOT = True

TRAIN_PROCESS_RLOO =False
TRAIN_OUTCOME_RLOO = False
TRAIN_PROCESS_RLOO_WITH_KL = False
TRAIN_OUTCOME_RLOO_WITH_KL = False
TRAIN_PROCESS_RLOO_WITH_KL_IS = False
TRAIN_OUTCOME_RLOO_WITH_KL_IS = False
TRAIN_PROCESS_RLOO_WITH_CLIPPING = False
TRAIN_OUTCOME_RLOO_WITH_CLIPPING = False
TRAIN_PROCESS_GRPO = False
TRAIN_OUTCOME_GRPO = True
TRAIN_PPO = True

PLOT_PROCESS_RLOO =True
PLOT_OUTCOME_RLOO = True
PLOT_PROCESS_RLOO_WITH_KL = True
PLOT_OUTCOME_RLOO_WITH_KL = True
PLOT_PROCESS_RLOO_WITH_KL_IS = True
PLOT_OUTCOME_RLOO_WITH_KL_IS = True
PLOT_PROCESS_RLOO_WITH_CLIPPING = True
PLOT_OUTCOME_RLOO_WITH_CLIPPING = True
PLOT_PROCESS_GRPO = True
PLOT_OUTCOME_GRPO = True
PLOT_PPO = True

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
    process_rloo_model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=process_rloo_eval_callback)
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
    outcome_rloo_model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=outcome_rloo_eval_callback)
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
    process_rloo_kl_model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=process_rloo_kl_eval_callback)
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
    outcome_rloo_kl_model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=outcome_rloo_kl_eval_callback)
    end_time = time.time()
    print(f"Outcome RLOO with KL training completed in {(end_time - start_time):.2f} seconds.")

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
    process_rloo_kl_is_model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=process_rloo_kl_is_eval_callback)
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
    outcome_rloo_kl_is_model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=outcome_rloo_kl_is_eval_callback)
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
    process_rloo_clipping_model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=process_rloo_clipping_eval_callback)
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
    outcome_rloo_clipping_model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=outcome_rloo_clipping_eval_callback)
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
        verbose=0,
        group_size=16,
        n_epochs=10,
    )
    print("Starting Process GRPO training...")
    start_time = time.time()
    grpo_model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=process_grpo_eval_callback)
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
        verbose=0,
        group_size=16,
        n_epochs=10,
        group_rollout_buffer_class=OutcomeGroupBuffer,
    )
    print("Starting Outcome GRPO training...")
    start_time = time.time()
    grpo_model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=outcome_grpo_eval_callback)
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
        batch_size=16,
        n_epochs=10,
    )
    print("Starting PPO training...")
    start_time = time.time()
    ppo_model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=ppo_eval_callback)
    end_time = time.time()
    print(f"PPO training completed in {(end_time - start_time):.2f} seconds.")

# --- Plotting Evaluation Results ---
if PLOT_PROCESS_RLOO:
    data = np.load(EVAL_PATH + "process_rloo/evaluations.npz")
    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    plt.plot(timesteps, mean_rewards, label="Process RLOO Mean Reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)

if PLOT_OUTCOME_RLOO:
    data = np.load(EVAL_PATH + "outcome_rloo/evaluations.npz")
    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    plt.plot(timesteps, mean_rewards, label="Outcome RLOO Mean Reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)

if PLOT_PROCESS_RLOO_WITH_KL:
    data = np.load(EVAL_PATH + "process_rloo_kl/evaluations.npz")
    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    plt.plot(timesteps, mean_rewards, label="Process RLOO with KL Mean Reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)

if PLOT_OUTCOME_RLOO_WITH_KL:
    data = np.load(EVAL_PATH + "outcome_rloo_kl/evaluations.npz")
    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    plt.plot(timesteps, mean_rewards, label="Outcome RLOO with KL Mean Reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)

if PLOT_PROCESS_RLOO_WITH_KL_IS:
    data = np.load(EVAL_PATH + "process_rloo_kl_is/evaluations.npz")
    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    plt.plot(timesteps, mean_rewards, label="Process RLOO with KL and IS Mean Reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)

if PLOT_OUTCOME_RLOO_WITH_KL_IS:
    data = np.load(EVAL_PATH + "outcome_rloo_kl_is/evaluations.npz")
    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    plt.plot(timesteps, mean_rewards, label="Outcome RLOO wuth KL and IS Mean Reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)

if PLOT_PROCESS_RLOO_WITH_CLIPPING:
    data = np.load(EVAL_PATH + "process_rloo_clipping/evaluations.npz")
    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    plt.plot(timesteps, mean_rewards, label="Process RLOO with Clipping Mean Reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)

if PLOT_OUTCOME_RLOO_WITH_CLIPPING:
    data = np.load(EVAL_PATH + "outcome_rloo_clipping/evaluations.npz")
    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    plt.plot(timesteps, mean_rewards, label="Outcome RLOO with Clipping Mean Reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)

if PLOT_PROCESS_GRPO:
    data = np.load(EVAL_PATH + "process_grpo/evaluations.npz")
    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    plt.plot(timesteps, mean_rewards, label="Process GRPO Mean Reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)

if PLOT_OUTCOME_GRPO:
    data = np.load(EVAL_PATH + "outcome_grpo/evaluations.npz")
    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    plt.plot(timesteps, mean_rewards, label="Outcome GRPO Mean Reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)

if PLOT_PPO:
    data = np.load(EVAL_PATH + "ppo/evaluations.npz")
    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    plt.plot(timesteps, mean_rewards, label="PPO Mean Reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)

plt.xlabel("Timesteps")
plt.ylabel("Evaluation Reward")
plt.title("Evaluation Performance Over Time")
plt.legend()
plt.grid()
plt.tight_layout()
if SAVE_EVAL_PLOT:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"./eval_images/eval_performance_{timestamp}.png")
plt.show()
