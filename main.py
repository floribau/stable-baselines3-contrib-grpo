"""Main module to run GRPO on CartPole-v1 environment."""
import warnings

import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.grpo.grpo import GRPO

warnings.filterwarnings("error", category=RuntimeWarning)  # DEBUG line for temporarily converting warnings to errors

vec_env = make_vec_env("CartPole-v1", n_envs=1)

model = GRPO("GroupPolicy", vec_env, verbose=1, group_size=16, learning_rate=0.001)
model.learn(total_timesteps=50_000)
# model.save("grpo_cartpole")

obs = vec_env.reset()
episode_rewards = [[] for _ in range(vec_env.num_envs)]
all_episode_rewards = []
MAX_TIMESTEPS = 10_000

for _ in range(MAX_TIMESTEPS):
    action, _ = model.predict(obs)
    obs, rewards, dones, _ = vec_env.step(action)
    # vec_env.render("human")

    for i in range(vec_env.num_envs):
        episode_rewards[i].append(rewards[i])
        if dones[i]:
            # Episode ended; store total reward
            total_reward = sum(episode_rewards[i])
            all_episode_rewards.append(total_reward)
            episode_rewards[i] = []
            # print(total_reward)

if all_episode_rewards:
    avg_reward = np.mean(all_episode_rewards)
    print(f"Average reward per episode over {len(all_episode_rewards)} episodes: {avg_reward:.2f}")
else:
    print("No episodes finished during eval.")
