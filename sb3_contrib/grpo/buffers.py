"""Module providing buffer class implementations for GRPO."""

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize


class Trajectory:
    """Class containing one trajectory of RL rollout steps."""

    def __init__(self, device: th.device | str = "auto", gamma: float = 1):
        self.device = get_device(device)
        self.gamma = gamma
        # IDEA use np.empty((0,), dtype=x) for better performance
        self.observations: list[np.ndarray] = []
        self.actions: list[int] = []  # IDEA use float for continuous actions
        self.rewards: list[float] = []
        self.log_probs: list[float] = []
        self.dones: list[bool] = []

    def add(self, obs: np.ndarray, action: int, reward: float, log_prob: float, done: bool):
        """Adds a single step to the trajectory.

        Args:
            obs (np.array): The observation
            action (int): The action
            reward (float): The reward
            log_prob (float): The log probability
            done (bool): The flag indicating whether the episode is done (terminated or truncated)
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get_return_sum(self) -> float:
        """
        Returns the trajectory return sum as the discounted sum of rewards.
        In standard GRPO, this is done without discounting (gamma=1).
        """
        return_sum = 0
        for r in reversed(self.rewards):
            return_sum = r + self.gamma * return_sum
        return return_sum

    def get_returns_to_go(self) -> np.ndarray[float]:
        """
        Returns the returns-to-go for each step as the discounted sum of rewards-to-go.
        In standard GRPO, this is done without discounting (gamma=1).
        """
        rollout_len = len(self.rewards)
        returns_to_go = np.empty(rollout_len)

        discounted_return = 0.0
        for t in reversed(range(rollout_len)):
            discounted_return = self.rewards[t] + self.gamma * discounted_return  # TODO check if reward is of correct dtype
            returns_to_go[t] = discounted_return

        return returns_to_go

    def to_tensor(self) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        observations = th.tensor(np.array(self.observations), device=self.device)
        actions = th.tensor(np.array(self.actions), device=self.device)
        log_probs = th.stack([th.as_tensor(p, device=self.device) for p in self.log_probs])  # IDEA use th.tensor directly?
        return observations, actions, log_probs


class GroupBuffer(BaseBuffer):
    """
    Buffer class containing a group of trajectories for a single GRPO update.
    """

    trajectories: list[Trajectory]
    returns: np.ndarray | None

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        scale_rewards: bool = False,
        device: th.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs)
        self.reset()
        self.scale_rewards = scale_rewards

    def add(self, traj: Trajectory) -> int:  # pylint: disable=arguments-differ
        """
        Adds the specified trajectory to the group trajectories if it is not full yet.
        Returns the pos of the trajectory in the buffer if added, otherwise returns -1.
        """
        if not self.full:
            pos = len(self.trajectories)  # len equals the pos index of the next element
            self.trajectories.append(traj)
            if len(self.trajectories) >= self.buffer_size:
                self.full = True
                self._compute_returns()
            return pos
        return -1

    def reset(self):
        super().reset()
        self.trajectories = []
        self.returns = None

    def _compute_returns(self):
        """
        Fills self.returns if not set yet.
        Return at index i corresponds to the trajectory at index i.
        """
        if self.returns is None:
            self.returns = np.array([traj.get_return_sum() for traj in self.trajectories])

    def get_advantages(self) -> np.ndarray:
        """
        Returns a numpy ndarray of advantages relative to the group.
        Advantage at index i corresponds to the trajectory at index i.
        """
        assert len(self.trajectories) > 0
        self._compute_returns()

        advantages = self.returns - np.mean(self.returns)
        if self.scale_rewards:
            advantages /= np.std(self.returns) + 1e-8  # Avoid division by zero

        return advantages

    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None):
        """
        Not used in GroupBuffer, but required by BaseBuffer.

        Raises:
            NotImplementedError: This method is not implemented for GRPO.
        """
        raise NotImplementedError
