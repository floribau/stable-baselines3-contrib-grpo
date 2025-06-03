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
        # IDEA use th.empty((0,), dtype=x) for better performance
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

    def get_returns_to_go(self) -> th.Tensor:
        """
        Returns the returns-to-go for each step as the discounted sum of rewards-to-go.
        In standard GRPO, this is done without discounting (gamma=1).
        """
        rollout_len = len(self.rewards)
        returns_to_go = th.empty(rollout_len)

        discounted_return = 0.0
        for t in reversed(range(rollout_len)):
            discounted_return = float(self.rewards[t]) + self.gamma * discounted_return
            returns_to_go[t] = discounted_return

        return returns_to_go

    def to_tensor(self) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        TODO docstring
        """
        observations_np = np.stack(self.observations)
        observations = th.as_tensor(observations_np, dtype=th.float32, device=self.device)

        actions_np = np.array(self.actions)
        actions = th.as_tensor(actions_np, dtype=th.float32, device=self.device)

        log_probs = th.as_tensor(self.log_probs, dtype=th.float32, device=self.device)

        return observations, actions, log_probs


class GroupBuffer(BaseBuffer):
    """
    Buffer class containing a group of trajectories for a single GRPO update.
    TODO write docstring for how advantage is calculated (all returns-to-go)
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
        Return at index t in a tensor i corresponds to the return for timestep t in trajectory i.
        """
        if self.returns is None:
            self.returns = [traj.get_returns_to_go() for traj in self.trajectories]

    def get_advantages(self) -> list[th.Tensor]:
        """
        Returns a list of advantage tensors relative to the group.
        Advantage at index t in a tensor i corresponds to the advantage for timestep t in trajectory i.
        """
        assert len(self.trajectories) > 0
        self._compute_returns()

        all_returns_to_go = th.cat(self.returns)
        mean_return = all_returns_to_go.mean()
        std_return = all_returns_to_go.std()

        advantages = [r - mean_return for r in self.returns]
        if self.scale_rewards:
            advantages = [a / (std_return + 1e-8) for a in advantages]  # avoid division by zero
        return advantages

    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None):
        """
        Not used in GroupBuffer, but required by BaseBuffer.

        Raises:
            NotImplementedError: This method is not implemented for GRPO.
        """
        raise NotImplementedError


class TimestepGroupBuffer(GroupBuffer):
    """
    TODO docstring how advantage is calculated (returns-to-go per timestep)
    """
    def _compute_returns(self):
        if self.returns is None:
            self.returns = [traj.get_returns_to_go() for traj in self.trajectories]

    def get_advantages(self) -> list[th.Tensor]:
        assert len(self.trajectories) > 0
        self._compute_returns()

        max_trajectory_length = max(len(traj.rewards) for traj in self.trajectories)
        advantages = [th.empty(len(traj_returns)) for traj_returns in self.returns]  # placeholder for advantages

        for t in range(max_trajectory_length):
            # BUG TypeError: only integer tensors of a single element can be converted to an index
            timestep_returns = np.ndarray([traj_returns[t] for traj_returns in self.returns if len(traj_returns) > t])
            mean_timestep_return = timestep_returns.mean()
            std_timestep_return = timestep_returns.std()

            for i, traj_returns in enumerate(self.returns):
                if len(traj_returns) > t:
                    single_advantage = traj_returns[t] - mean_timestep_return
                    if self.scale_rewards:
                        single_advantage /= std_timestep_return
                    advantages[i][t] = single_advantage

            return advantages