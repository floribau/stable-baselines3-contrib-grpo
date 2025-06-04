"""GRPO module written by ChatGPT"""

import sys
import time
import warnings
from collections import deque
from copy import deepcopy
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common import utils
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import ConstantSchedule, obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

from sb3_contrib.grpo.buffers import GroupBuffer, Trajectory
from sb3_contrib.grpo.policies import ActorPolicy

SelfGRPO = TypeVar("SelfGRPO", bound="GRPO")


class GRPO(BaseAlgorithm):
    """
    TODO write docstring
    """

    group_rollout_buffer: GroupBuffer
    policy: ActorPolicy
    policy_ref: ActorPolicy | None

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorPolicy,
        "ActorPolicy": ActorPolicy,
        "GroupPolicy": ActorPolicy,
    }

    def __init__(
        self,
        policy: str | type[ActorPolicy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        group_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 1,
        clip_range: float | Schedule = 0.2,
        scale_rewards: bool = False,
        kl_beta: float = 0.01,
        ent_coef: float = 0.005,
        max_grad_norm: float | None = 0.5,
        use_sde: bool = False,  # seems not to be relevant if spaces.Box is not supported as action space
        sde_sample_freq: int = -1,  # seems not to be relevant if spaces.Box is not supported as action space
        group_rollout_buffer_class: type[GroupBuffer] | None = None,
        group_rollout_buffer_kwargs: dict[str, Any] | None = None,
        stats_window_size: int = 100,
        tensorboard_log: bool = None,
        monitor_wrapper: bool = True,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=False,  # NOTE currently, GRPO only supports single env
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=(spaces.Discrete),
        )
        self.group_size = group_size
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.scale_rewards = scale_rewards
        self.kl_beta = kl_beta
        self.ent_coef = ent_coef
        assert max_grad_norm is None or max_grad_norm > 0, "max_grad_norm must be None or a positive float."
        self.max_grad_norm = max_grad_norm
        self.group_rollout_buffer_class = group_rollout_buffer_class
        self.group_rollout_buffer_kwargs = group_rollout_buffer_kwargs or {}

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.group_rollout_buffer_class is None:
            self.group_rollout_buffer_class = GroupBuffer

        self.group_rollout_buffer = self.group_rollout_buffer_class(
            buffer_size=self.group_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            scale_rewards=self.scale_rewards,
            device=self.device,
            n_envs=self.n_envs,
            **self.group_rollout_buffer_kwargs,
        )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)
        self.policy_ref = None
        self.clip_range = ConstantSchedule(self.clip_range)
        # Warn when not using CPU with MlpPolicy
        self._maybe_recommend_cpu()

    def _maybe_recommend_cpu(self, mlp_class_name: str = "GRPOPolicy") -> None:
        """
        Recommend to use CPU only when using GRPO with GRPOPolicy.

        :param: The name of the class for the default MlpPolicy.
        """
        policy_class_name = self.policy_class.__name__
        if self.device != th.device("cpu") and policy_class_name == mlp_class_name:
            warnings.warn(
                f"You are trying to run {self.__class__.__name__} on the GPU, "
                "but it is primarily intended to run on the CPU when not using a CNN policy "
                f"(you are using {policy_class_name} which should be a MlpPolicy). "
                "See https://github.com/DLR-RM/stable-baselines3/issues/1245 "
                "for more info. "
                "You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU."
                "Note: The model will train, but the GPU utilization will be poor and "
                "the training might take longer than on CPU.",
                UserWarning,
            )

    def collect_group_rollouts(self, env: VecEnv, callback: BaseCallback, group_size: int):
        """Collect a group of rollouts from the current policy and returns it as a group buffer."""
        assert self.group_rollout_buffer is not None, "Group rollout buffer must be initialized before collecting rollouts."

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        initial_obs = env.reset()
        initial_env = deepcopy(env)

        for _ in range(group_size):
            # TODO is there a better option than deepcopying?
            env = deepcopy(initial_env)  # copying to start rollouts from the same state
            last_obs = initial_obs  # TODO use self._last_obs instead
            traj = Trajectory()
            dones = False

            callback.on_rollout_start()

            while not dones:
                with th.no_grad():
                    obs_tensor = obs_as_tensor(last_obs, self.device)
                    actions, log_probs = self.policy(obs_tensor)  # plural naming convention for multiple parallel envs
                actions = actions.cpu().numpy()  # convert tensor to numpy array since env.step requires numpy array as input
                next_obs, rewards, dones, infos = env.step(actions)

                callback.update_locals(locals())
                callback.on_step()

                self.num_timesteps += env.num_envs

                self._update_info_buffer(infos, dones)

                if isinstance(self.action_space, spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = actions.reshape(-1, 1)

                # HACK the float cast is only a quickfix, more work needs to be done for multiple parallel envs
                traj.add(obs=last_obs, action=actions, reward=float(rewards), log_prob=log_probs, done=dones)
                last_obs = next_obs

            self.group_rollout_buffer.add(traj)

            callback.update_locals(locals())
            callback.on_rollout_end()

    def train(self) -> None:
        """Update policy params."""
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)

        pg_losses, kl_losses, entropy_losses, clip_fractions, loss = self._train_process_supervision(clip_range)

        # Logs
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/kl_loss", np.mean(kl_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/total_loss", loss.item())  # IDEA it probably makes more sense to use mean of a list here
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

    def _train_process_supervision(self, clip_range: float) -> tuple[list, list, list, list, th.Tensor]:
        pg_losses, kl_losses, entropy_losses, clip_fractions = [], [], [], []

        advantages = self.group_rollout_buffer.get_advantages()

        for _ in range(self.n_epochs):

            for traj_idx, traj in enumerate(self.group_rollout_buffer.trajectories):
                obs, actions, old_log_probs = traj.to_tensor()
                old_log_probs = old_log_probs.detach()

                if len(obs) == 0:
                    continue

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = actions.long().flatten()

                current_log_probs, entropy = self.policy.evaluate_actions(obs, actions)

                ratios = th.exp(current_log_probs - old_log_probs)

                advantage_tensor = advantages[traj_idx].detach()

                # Surrogate loss
                surr1 = ratios * advantage_tensor
                surr2 = th.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * advantage_tensor
                policy_loss = -th.min(surr1, surr2).mean()

                # KL divergence penalty
                with th.no_grad():
                    log_probs_ref, _ = self.policy_ref.evaluate_actions(obs, actions)
                kl_ratios = log_probs_ref - current_log_probs.detach()
                kl_div_estimate = th.exp(kl_ratios) - kl_ratios - 1
                kl_loss = kl_div_estimate.mean()

                # Entropy loss
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-current_log_probs)
                else:
                    entropy_loss = -th.mean(entropy)

                loss = policy_loss + self.kl_beta * kl_loss + self.ent_coef * entropy_loss

                # Logging
                pg_losses.append(policy_loss.item())
                kl_losses.append(kl_loss.item())
                entropy_losses.append(entropy_loss.item())
                clip_fraction = th.mean((th.abs(ratios - 1.0) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                self.policy.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1  # this is in accordance with PPO from SB3

        return pg_losses, kl_losses, entropy_losses, clip_fractions, loss

    def dump_logs(self, iteration: int = 0):
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
            self.logger.record("time/trajectories", iteration * self.group_size, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 10,
        tb_log_name: str = "GRPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "GRPO":
        total_timesteps, callback = self._setup_learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=tb_log_name,
            progress_bar=progress_bar,
        )

        callback.on_training_start(locals(), globals())
        assert self.env is not None, "Environment must be set before calling learn()"
        assert total_timesteps >= self.group_size, f"At least {self.group_size} must be collected because of group size."

        iteration = 0
        while self.num_timesteps < total_timesteps:
            self.group_rollout_buffer.reset()  # Reset the group buffer before collecting new rollouts
            self.collect_group_rollouts(env=self.env, callback=callback, group_size=self.group_size)

            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if self._current_progress_remaining < 0:
                # self.num_timesteps > total_timesteps: stop training
                break

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None, "Episode info buffer must be initialized before logging."
                self.dump_logs(iteration)

            self.policy_ref = self.policy.get_frozen_deepcopy()  # set reference policy to the current policy
            self.train()  # Update the policy params based on the collected group rollouts

            iteration += 1

        return self

    def _setup_learn(
        self,
        # total_timesteps: int,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ):
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0  # TODO check what this does
        else:
            # Make sure training trajectories are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # TODO avoid resetting the env when calling .learn() consecutive times

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback
