from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


ENV_NAME = "Acrobot-v1"
NORMALIZE_ENV = True
N_TIMESTEPS = int(1e6)
EVAL_FREQ = 5000
N_EVAL_EPISODES = 4
EVAL_PLOT_OPAQUENESS_ALPHA = 0.2
EVAL_LOGS_PATH = f"./eval/eval_logs/tuned_params/{ENV_NAME}/"
MODELS_PATH = f"./eval/models/tuned_params/{ENV_NAME}/"
EVAL_PLOTS_PATH = f"./eval/eval_plots/tuned_params/{ENV_NAME}/"

TRAIN_PPO = False
TRAIN_TUNED_PPO = True
TRAIN_FAIR_TUNED_PPO = False

PLOT_PPO = True
PLOT_TUNED_PPO = True
PLOT_FAIR_TUNED_PPO = False

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

default_env = make_vec_env(ENV_NAME, n_envs=1)
tuned_env = make_vec_env(ENV_NAME, n_envs=16)
eval_env = make_vec_env(ENV_NAME, n_envs=1)
if NORMALIZE_ENV:
    default_env = VecNormalize(default_env, norm_obs=True, norm_reward=False)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    tuned_env = VecNormalize(tuned_env, norm_obs=True, norm_reward=False)

default_ppo_model = PPO("MlpPolicy", default_env, verbose=1)
tuned_ppo_model = PPO(
    "MlpPolicy",
    tuned_env,
    verbose=1,
    n_steps=256,
    gae_lambda=0.94,
    gamma=0.99,
    n_epochs=4,
    ent_coef=0.0,
)
fair_tuned_ppo_model = PPO(
    "MlpPolicy",
    default_env,
    verbose=1,
    n_steps=256,
    gae_lambda=0.94,
    gamma=0.99,
    n_epochs=4,
    ent_coef=0.0,
)

eval_callback = EvalCallback(
    eval_env,
    log_path=EVAL_LOGS_PATH + "ppo/",
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
    verbose=0,
)
tuned_eval_callback = EvalCallback(
    eval_env,
    log_path=EVAL_LOGS_PATH + "tuned_ppo/",
    eval_freq=EVAL_FREQ / tuned_env.num_envs,
    # eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
    verbose=0,
)
fair_tuned_eval_callback = EvalCallback(
    eval_env,
    log_path=EVAL_LOGS_PATH + "fair_tuned_ppo/",
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
    verbose=0,
)

# --- Train and save the models ---
if TRAIN_PPO:
    default_ppo_model.learn(total_timesteps=N_TIMESTEPS, callback=eval_callback)
    default_ppo_model.save(f"{MODELS_PATH}ppo_model")

if TRAIN_TUNED_PPO:
    tuned_ppo_model.learn(total_timesteps=N_TIMESTEPS, callback=tuned_eval_callback)
    tuned_ppo_model.save(f"{MODELS_PATH}tuned_ppo_model")

if TRAIN_FAIR_TUNED_PPO:
    fair_tuned_ppo_model.learn(total_timesteps=N_TIMESTEPS, callback=fair_tuned_eval_callback)
    fair_tuned_ppo_model.save(f"{MODELS_PATH}fair_tuned_ppo_model")

# --- Plot the PPO eval results ---
if PLOT_PPO:
    data = np.load(EVAL_LOGS_PATH + "ppo/evaluations.npz")
    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    plt.plot(timesteps, mean_rewards, label="Default PPO Reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

# --- Plot the tuned PPO eval results ---
if PLOT_TUNED_PPO:
    data = np.load(EVAL_LOGS_PATH + "tuned_ppo/evaluations.npz")
    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    plt.plot(timesteps, mean_rewards, label="Tuned PPO Reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

# --- Plot the fair tuned PPO eval results ---
if PLOT_FAIR_TUNED_PPO:
    data = np.load(EVAL_LOGS_PATH + "fair_tuned_ppo/evaluations.npz")
    timesteps = data["timesteps"]
    results = data["results"]

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    plt.plot(timesteps, mean_rewards, label="Fair Tuned PPO Reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=EVAL_PLOT_OPAQUENESS_ALPHA)

# --- Finalize the plot ---
plt.xlabel("Timesteps")
plt.ylabel("Evaluation Reward")
plt.title(f"{ENV_NAME} Evaluation Performance Over Time")
plt.legend(loc="best", framealpha=0.3)
plt.grid()
plt.tight_layout()
os.makedirs(EVAL_PLOTS_PATH, exist_ok=True)
plt.savefig(f"{EVAL_PLOTS_PATH}eval_performance_{ENV_NAME}_{timestamp}.png")
plt.show()
