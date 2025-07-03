"""Module providing buffer class implementations for GRPO."""

from enum import Enum

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize


class SupervisionType(Enum):
    """Enum for the type of supervision used in GRPO."""

    OUTCOME = 0
    PROCESS = 1

    def is_outcome_supervision(self) -> bool:
        """Returns True if the supervision type is outcome supervision."""
        return self == SupervisionType.OUTCOME

    def is_process_supervision(self) -> bool:
        """Returns True if the supervision type is process supervision."""
        return self == SupervisionType.PROCESS


class Trajectory:
    """Class containing one trajectory of RL rollout steps."""

    observations: list[np.ndarray]
    actions: list[int]  # IDEA use float instead to support continuous action spaces
    rewards: list[float]
    log_probs: list[float]
    dones: list[bool]

    def __init__(
        self,
        device: th.device | str = "auto",
        gamma: float = 1,
    ):
        self.device = get_device(device)
        self.gamma = gamma

        # IDEA use th.empty((0,), dtype=x) for better performance
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []

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
            assert isinstance(self.rewards[t], (float, int)), "Reward at timestep t should be a number."
            discounted_return = self.rewards[t] + self.gamma * discounted_return
            returns_to_go[t] = discounted_return

        return returns_to_go

    def to_tensor(self) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Returns the observations, actions, and log probabilities as tensors.
        """
        observations_np = np.stack(self.observations)
        observations = th.from_numpy(observations_np).float().to(self.device)

        actions_np = np.array(self.actions)
        actions = th.from_numpy(actions_np).to(self.device)

        log_probs = th.tensor(self.log_probs, dtype=th.float32, device=self.device)

        return observations, actions, log_probs


class GroupBuffer(BaseBuffer):
    """
    Base GRPO buffer class containing a group of trajectories for a single GRPO update.
    """

    trajectories: list[Trajectory]
    returns: list[th.Tensor] | None
    supervision_type: SupervisionType

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

    def reset(self):
        super().reset()
        self.trajectories = []
        self.returns = None

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
                self._maybe_compute_returns()
            return pos
        return -1

    def _maybe_compute_returns(self):
        """
        Fills self.returns if not set yet.
        Return at index t in a tensor i corresponds to the return for timestep t in trajectory i
        (if implementing process supervision).
        """
        raise NotImplementedError

    def get_advantages(self) -> list[th.Tensor]:
        """
        Returns a list of advantage tensors relative to the group. This is the returns of all trajectories are
        subtracted as a baseline and possibly the returns are scaled by the standard deviation.
        Advantage at index t in a tensor i corresponds to the advantage for timestep t in trajectory i
        (if implementing process supervision).
        """
        raise NotImplementedError

    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None):
        """
        Not used in GroupBuffer, but required by BaseBuffer.

        Raises:
            NotImplementedError: This method is not implemented for GRPO.
        """
        raise NotImplementedError


class TimestepGroupBuffer(GroupBuffer):
    """
    Buffer class containing a group of trajectories for a single GRPO update.
    The buffer implements process supervision, where the advantage is computed as the returns-to-go at each timestep t relative
    to the mean return-to-go of all trajectories at timestep t in the group.

    BUG this version doesn't work because the update signal is too small.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        scale_rewards: bool = False,
        device: th.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, scale_rewards, device, n_envs)
        self.supervision_type = SupervisionType.PROCESS

    def _maybe_compute_returns(self):
        if self.returns is None:
            self.returns = [traj.get_returns_to_go() for traj in self.trajectories]

    def get_advantages(self) -> list[th.Tensor]:
        assert len(self.trajectories) > 0
        self._maybe_compute_returns()

        max_trajectory_length = max(traj_returns.size(0) for traj_returns in self.returns)
        advantages = [th.empty(traj_returns.size(0)) for traj_returns in self.returns]  # placeholder for advantages

        for t in range(max_trajectory_length):
            timestep_returns = np.array([traj_returns[t] if len(traj_returns) > t else 0 for traj_returns in self.returns])
            mean_timestep_return = timestep_returns.mean()
            std_timestep_return = timestep_returns.std()

            for i, traj_returns in enumerate(self.returns):
                if len(traj_returns) > t:
                    single_advantage = traj_returns[t] - mean_timestep_return
                    if self.scale_rewards:
                        single_advantage /= std_timestep_return + 1e-8  # avoid division by zero
                    advantages[i][t] = single_advantage

            return advantages


class ProcessGroupBuffer(GroupBuffer):
    """
    Buffer class containing a group of trajectories for a single GRPO update.
    The buffer implements process supervision, where the advantage is computed as the returns-to-go at each timestep relative
    to the mean return-to-go of all timesteps in the group.

    The idea for this implementation has been taken from Emanuel Ruzak (https://github.com/emparu/PPO-vs-GRPO)
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        scale_rewards: bool = False,
        device: th.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, scale_rewards, device, n_envs)
        self.supervision_type = SupervisionType.PROCESS

    def _maybe_compute_returns(self):
        if self.returns is None:
            self.returns = [traj.get_returns_to_go() for traj in self.trajectories]

    def get_advantages(self) -> list[th.Tensor]:
        assert len(self.trajectories) > 0, "Cannot compute advantages: No trajectories in the buffer."
        self._maybe_compute_returns()

        all_returns_to_go = th.cat(self.returns)
        mean_return = all_returns_to_go.mean()
        std_return = all_returns_to_go.mean()

        advantages = [r - mean_return for r in self.returns]
        if self.scale_rewards:
            advantages = [a / (std_return + 1e-8) for a in advantages]  # avoid division by zero
        return advantages


class DeepSeekProcessGroupBuffer(GroupBuffer):
    """
    Buffer class containing a group of trajectories for a single GRPO update.
    The buffer implements process supervision, where the advantage is computed as the returns-to-go with the per-step rewards
    normalized by the average per-step reward of all steps in the trajectory.

    This implementation conforms to Process Supervision in DeepSeekMath (https://arxiv.org/pdf/2402.03300).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        scale_rewards: bool = False,
        device: th.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, scale_rewards, device, n_envs)
        self.supervision_type = SupervisionType.PROCESS

    def _maybe_compute_returns(self):
        if self.returns is None:
            self.returns = [th.tensor(traj.rewards, dtype=th.float32, device=self.device) for traj in self.trajectories]

    def get_advantages(self) -> list[th.Tensor]:
        self._maybe_compute_returns()

        all_returns = th.cat(self.returns)
        mean_returns = all_returns.mean()
        std_returns = all_returns.std()

        normalized_returns = [traj_returns - mean_returns for traj_returns in self.returns]
        if self.scale_rewards:
            normalized_returns = [traj_returns / (std_returns + 1e-8) for traj_returns in normalized_returns]

        advantages = [
            th.flip(th.cumsum(th.flip(r, dims=[0]), dim=0), dims=[0]) for r in normalized_returns
        ]  # advantage at every step t is return to go of normalized returns starting from t
        return advantages


class DeepSeekOutcomeGroupBuffer(GroupBuffer):
    """
    Buffer class containing a group of trajectories for a single GRPO update.
    The buffer implements outcome supervision, where the advantage is computed as the trajectory reward relative
    to the mean trajectory reward of all trajectories in the group.
    The trajectory reward is the sum of all rewards in the trajectory.

    This implementation conforms to Outcome Supervision in DeepSeekMath (https://arxiv.org/pdf/2402.03300).
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        scale_rewards: bool = False,
        device: th.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, scale_rewards, device, n_envs)
        self.supervision_type = SupervisionType.OUTCOME

    def _maybe_compute_returns(self):
        """
        See super._maybe_compute_returns() for full documentation.
        This works similarly, but computes per-trajectories returns instead of per-step returns.
        """
        if self.returns is None:
            # list of scalar tensors
            self.returns = [th.tensor(sum(traj.rewards), dtype=th.float32, device=self.device) for traj in self.trajectories]

    def get_advantages(self) -> list[th.Tensor]:
        """
        See super.get_advantages() for full documentation.
        This works similarly, but computes per-trajectory advantages instead of per-step advantages.
        """
        self._maybe_compute_returns()
        print(self.returns)
        mean_return = th.stack(self.returns).mean()
        std_return = th.stack(self.returns).std()

        advantages = [r - mean_return for r in self.returns]
        if self.scale_rewards:
            advantages = [a / (std_return + 1e-8) for a in advantages]  # avoid division by zero
        return advantages

    def get_leave_one_out_advantages(self) -> list[th.Tensor]:
        """
        Similar to get_advantages() but uses a leave-one-out baseline, where the returns of all trajectories except
        trajectory T are used to compute the baseline for trajectory T.
        """
        assert len(self.trajectories) > 0, "No trajectories in the buffer to compute advantages from."
        self._maybe_compute_returns()

        total_return_sum = th.stack(self.returns).sum()

        k = len(self.returns)
        advantages = [
            r - (total_return_sum - r) / (k - 1)
            for r in self.returns
        ]
        return advantages


BUFFER_CLASS_ALIASES = {
    "GroupBuffer": GroupBuffer,
    "TimestepGroupBuffer": TimestepGroupBuffer,
    "ProcessGroupBuffer": ProcessGroupBuffer,
    "DeepSeekProcessGroupBuffer": DeepSeekProcessGroupBuffer,
    "DeepSeekOutcomeGroupBuffer": DeepSeekOutcomeGroupBuffer,
    "OutcomeGroupBuffer": DeepSeekOutcomeGroupBuffer,  # alias for DeepSeekOutcomeGroupBuffer
}
