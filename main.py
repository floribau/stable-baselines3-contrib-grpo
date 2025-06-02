"""Main module to run GRPO on CartPole-v1 environment."""
import warnings

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo import PPO
from sb3_contrib.grpo.grpo import GRPO

warnings.filterwarnings("error", category=RuntimeWarning)  # DEBUG line for temporarily converting warnings to errors

TRAINING_TIMESTEPS = 50_000
EVAL_TIMESTEPS = 10_000

grpo_vec_env = make_vec_env("CartPole-v1", n_envs=1)
ppo_vec_env = make_vec_env("CartPole-v1", n_envs=1)

grpo_model = GRPO("GroupPolicy", grpo_vec_env, verbose=1, group_size=16, learning_rate=0.001)
grpo_model.learn(total_timesteps=TRAINING_TIMESTEPS)
ppo_model = PPO("MlpPolicy", ppo_vec_env, verbose=0, learning_rate=0.001, batch_size=16)
ppo_model.learn(total_timesteps=TRAINING_TIMESTEPS)
# model.save("grpo_cartpole")

obs = grpo_vec_env.reset()
episode_rewards = [[] for _ in range(grpo_vec_env.num_envs)]
all_episode_rewards = []

# GRPO eval
for _ in range(EVAL_TIMESTEPS):
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

obs = grpo_vec_env.reset()
episode_rewards = [[] for _ in range(grpo_vec_env.num_envs)]
all_episode_rewards = []

# PPO eval as baseline
for _ in range(EVAL_TIMESTEPS):
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
