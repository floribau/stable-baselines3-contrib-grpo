"""Main module to run GRPO on CartPole-v1 environment."""
from datetime import datetime
import warnings

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.ppo import PPO
from sb3_contrib.grpo.grpo import GRPO

warnings.filterwarnings("error", category=RuntimeWarning)  # DEBUG line for temporarily converting warnings to errors

TRAINING_TIMESTEPS = 40_000
RUN_TIMESTEPS = 100

RUN_GRPO = False
RUN_PPO = False

N_EVAL_EPISODES = 5
EVAL_FREQ = 2000
GRPO_EVAL_PATH = "./eval_logs/grpo/"
PPO_EVAL_PATH = "./eval_logs/ppo/"

grpo_vec_env = make_vec_env("CartPole-v1", n_envs=1)
ppo_vec_env = make_vec_env("CartPole-v1", n_envs=1)
grpo_eval_env = make_vec_env("CartPole-v1", n_envs=1)
ppo_eval_env = make_vec_env("CartPole-v1", n_envs=1)

grpo_eval_callback = EvalCallback(
    grpo_eval_env,
    log_path=GRPO_EVAL_PATH,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
    verbose=0,
)
ppo_eval_callback = EvalCallback(
    ppo_eval_env,
    log_path=PPO_EVAL_PATH,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    render=False,
    verbose=0,
)

grpo_model = GRPO(
    "GroupPolicy",
    grpo_vec_env,
    verbose=1,
    group_size=16,
    learning_rate=0.001,
    kl_beta=0,
)
# grpo_model = GRPO("GroupPolicy", grpo_vec_env, verbose=1, group_size=16, learning_rate=0.001, kl_beta=0)
grpo_model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=grpo_eval_callback)

ppo_model = PPO("MlpPolicy", ppo_vec_env, verbose=1, learning_rate=0.001, batch_size=16)
ppo_model.learn(total_timesteps=TRAINING_TIMESTEPS, callback=ppo_eval_callback)

if RUN_GRPO:
    obs = grpo_vec_env.reset()
    episode_rewards = [[] for _ in range(grpo_vec_env.num_envs)]
    all_episode_rewards = []

    for _ in range(RUN_TIMESTEPS):
        action, _ = grpo_model.predict(obs)
        obs, rewards, dones, _ = grpo_vec_env.step(action)
        # vec_env.render("human")

        for i in range(grpo_vec_env.num_envs):
            episode_rewards[i].append(rewards[i])
            if dones[i]:
                # Episode ended; store total reward
                total_reward = sum(episode_rewards[i])
                all_episode_rewards.append(total_reward)
                episode_rewards[i] = []
                # print(total_reward)

    if all_episode_rewards:
        avg_reward = np.mean(all_episode_rewards)
        print(f"GRPO: average reward per episode over {len(all_episode_rewards)} episodes: {avg_reward:.2f}")
    else:
        print("No episodes finished during GRPO eval.")

if RUN_PPO:
    obs = grpo_vec_env.reset()
    episode_rewards = [[] for _ in range(grpo_vec_env.num_envs)]
    all_episode_rewards = []

    for _ in range(RUN_TIMESTEPS):
        action, _ = ppo_model.predict(obs)
        obs, rewards, dones, _ = ppo_vec_env.step(action)

        for i in range(ppo_vec_env.num_envs):
            episode_rewards[i].append(rewards[i])
            if dones[i]:
                total_reward = sum(episode_rewards[i])
                all_episode_rewards.append(total_reward)
                episode_rewards[i] = []

    if all_episode_rewards:
        avg_reward = np.mean(all_episode_rewards)
        print(f"PPO: average reward per episode over {len(all_episode_rewards)} episodes: {avg_reward:.2f}")
    else:
        print("No episodes finished during PPO eval.")

grpo_data = np.load(GRPO_EVAL_PATH + "evaluations.npz")
grpo_timesteps = grpo_data["timesteps"]
grpo_results = grpo_data["results"]

grpo_mean_rewards = grpo_results.mean(axis=1)
grpo_std_rewards = grpo_results.std(axis=1)

ppo_data = np.load(PPO_EVAL_PATH + "evaluations.npz")
ppo_timesteps = ppo_data["timesteps"]
ppo_results = ppo_data["results"]

ppo_mean_rewards = ppo_results.mean(axis=1)
ppo_std_rewards = ppo_results.std(axis=1)

plt.plot(grpo_timesteps, grpo_mean_rewards, label="GRPO Mean Evaluation Reward")
plt.fill_between(grpo_timesteps, grpo_mean_rewards - grpo_std_rewards, grpo_mean_rewards + grpo_std_rewards, alpha=0.3)
plt.plot(ppo_timesteps, ppo_mean_rewards, label="PPO Mean Evaluation Reward")
plt.fill_between(ppo_timesteps, ppo_mean_rewards - ppo_std_rewards, ppo_mean_rewards + ppo_std_rewards, alpha=0.3)
plt.xlabel("Timesteps")
plt.ylabel("Evaluation Reward")
plt.title("Evaluation Performance Over Time")
plt.legend()
plt.grid()
plt.tight_layout()
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plt.savefig(f"./eval_images/eval_performance_{timestamp}.png")
plt.show()
