"""Module containing the GRPO class"""

import sys
import time
import warnings
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import ConstantSchedule, obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

from sb3_contrib.common.buffers import GroupBuffer, ProcessGroupBuffer, SupervisionType, Trajectory
from sb3_contrib.grpo.policies import ActorPolicy

SelfGRPO = TypeVar("SelfGRPO", bound="GRPO")


class GRPO(BaseAlgorithm):
    """
    Group Relative Policy Optimization algorithm (GRPO)

    Paper: https://arxiv.org/pdf/2402.03300
    Code: This implementation borrows code ideas from Emanuel Ruzak (https://github.com/emparu/PPO-vs-GRPO)

    Introduction to GRPO: https://huggingface.co/docs/trl/main/en/grpo_trainer

    :param policy: The policy model to use (MlpPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param group_size: The number of trajectories to collect in a group.
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor. In standard GRPO, it is set to 1.
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param batch_group_updates: Whether the policy should be updated with batch updates.
        If ``True``, one batch update is performed for the whole group of trajectories.
        If ``False``, one update is performed per trajectory.
    :param scale_rewards: Whether to scale rewards by the standard deviation during advantage calculation.
    :param kl_beta: KL divergence penalty coefficient for the loss calculation.
    :param kl_ref_iterations: Number of learning iterations before updating the reference policy for KL divergence penalty.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param supervision_type: Type of supervision to use for the group rollout buffer.
    :param group_rollout_buffer_class: Group Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param group_rollout_buffer_kwargs: Keyword arguments to pass to the group rollout buffer on creation
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ppo_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
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
        group_size: int = 64,  # IDEA use n_envs for group_size, making it more efficient due to parallelization
        n_epochs: int = 10,
        gamma: float = 1,
        clip_range: float | Schedule = 0.2,
        batch_group_updates: bool = False,
        scale_rewards: bool = False,
        kl_beta: float = 0.01,
        kl_ref_iterations: int = 1,
        ent_coef: float = 0.005,
        max_grad_norm: float | None = 0.5,
        use_sde: bool = False,  # seems not to be relevant unless spaces.Box is supported as action space
        sde_sample_freq: int = -1,  # seems not to be relevant unless spaces.Box is supported as action space
        supervision_type: SupervisionType = SupervisionType.PROCESS,
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
            supported_action_spaces=(spaces.Discrete),  # TODO implement Box action spaces (and possibly MultiDiscrete)
        )
        self.group_size = group_size
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.batch_group_updates = batch_group_updates
        self.scale_rewards = scale_rewards
        self.kl_beta = kl_beta
        self.kl_ref_iterations = kl_ref_iterations
        self.ent_coef = ent_coef
        assert max_grad_norm is None or max_grad_norm > 0, "max_grad_norm must be None or a positive float."
        self.max_grad_norm = max_grad_norm
        self.supervision_type = supervision_type
        self.group_rollout_buffer_class = group_rollout_buffer_class
        self.group_rollout_buffer_kwargs = group_rollout_buffer_kwargs or {}

        if _init_setup_model:
            self._setup_model()
        assert self.supervision_type == self.group_rollout_buffer_class.supervision_type

    def _setup_model(self):
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)  # TODO this needs to be adjusted (not really used for env creation)

        if self.group_rollout_buffer_class is None:
            self.group_rollout_buffer_class = ProcessGroupBuffer

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

        random_seed = np.random.randint(0, 2**32 - 1)

        for _ in range(group_size):
            env.seed(random_seed)
            self._last_obs = env.reset()  # always reset envs to the same state in the same group

            traj = Trajectory(device=self.device, gamma=self.gamma)

            dones = False

            callback.on_rollout_start()

            while not dones:
                with th.no_grad():
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
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
                traj.add(obs=self._last_obs, action=actions, reward=float(rewards), log_prob=log_probs, done=dones)
                self._last_obs = next_obs

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

        if self.group_rollout_buffer.supervision_type.is_process_supervision():
            pg_losses, kl_losses, entropy_losses, clip_fractions, losses = self._train_process_supervision(clip_range)
        else:
            pg_losses, kl_losses, entropy_losses, clip_fractions, losses = self._train_outcome_supervision(clip_range)

        # Logs
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/kl_loss", np.mean(kl_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/total_loss", np.mean(losses))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

    def _train_process_supervision(self, clip_range: float) -> tuple[list, list, list, list, list]:
        """
        Process supvervision training method.
        """
        assert self.group_rollout_buffer.supervision_type.is_process_supervision(), "Buffer must be process supervision type."
        # IDEA factor out policy updating, this is equal for all supervision types
        pg_losses, kl_losses, entropy_losses, clip_fractions, losses = [], [], [], [], []

        advantages = self.group_rollout_buffer.get_advantages()

        for _ in range(self.n_epochs):

            obs_list, actions_list, old_log_probs_list, advantages_list = [], [], [], []

            for traj_idx, traj in enumerate(self.group_rollout_buffer.trajectories):
                obs, actions, old_log_probs = traj.to_tensor()

                old_log_probs = old_log_probs.detach()

                if len(obs) == 0:
                    continue

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = actions.long().flatten()

                advantage_tensor = advantages[traj_idx].detach()

                if self.batch_group_updates:
                    # Only collect values for batched policy update for the whole group
                    obs_list.append(obs)
                    actions_list.append(actions)
                    old_log_probs_list.append(old_log_probs)
                    advantages_list.append(advantage_tensor)

                else:
                    # Policy update for each trajectory separately
                    current_log_probs, entropy = self.policy.evaluate_actions(obs, actions)

                    # --- Policy Gradient loss ---
                    ratios = th.exp(current_log_probs - old_log_probs)
                    surr1 = ratios * advantage_tensor
                    surr2 = th.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * advantage_tensor
                    policy_loss = -th.min(surr1, surr2).mean()

                    # --- KL loss ---
                    with th.no_grad():
                        log_probs_ref, _ = self.policy_ref.evaluate_actions(obs, actions)
                    kl_ratios = log_probs_ref - current_log_probs
                    kl_div_estimate = th.exp(kl_ratios) - kl_ratios - 1
                    kl_loss = kl_div_estimate.mean()

                    # --- Entropy loss ---
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-current_log_probs)
                    else:
                        entropy_loss = -th.mean(entropy)

                    # --- Total loss ---
                    loss = policy_loss + self.kl_beta * kl_loss + self.ent_coef * entropy_loss

                    # --- Logging ---
                    pg_losses.append(policy_loss.item())
                    kl_losses.append(kl_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    losses.append(loss.item())
                    clip_fraction = th.mean((th.abs(ratios - 1.0) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    # --- Backprop ---
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    if self.max_grad_norm is not None:
                        # Clip grad norm
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

            if self.batch_group_updates:
                # Finally the batched policy update for the whole group
                obs = th.cat(obs_list, dim=0)
                actions = th.cat(actions_list, dim=0)
                old_log_probs = th.cat(old_log_probs_list, dim=0)
                advantages_tensor = th.cat(advantages_list, dim=0)

                current_log_probs, entropy = self.policy.evaluate_actions(obs, actions)

                # --- Policy Gradient loss ---
                ratios = th.exp(current_log_probs - old_log_probs)
                surr1 = ratios * advantages_tensor
                surr2 = th.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * advantages_tensor
                policy_loss = -th.min(surr1, surr2).mean()

                # --- KL loss ---
                with th.no_grad():
                    log_probs_ref, _ = self.policy_ref.evaluate_actions(obs, actions)
                kl_ratios = log_probs_ref - current_log_probs
                kl_div_estimate = th.exp(kl_ratios) - kl_ratios - 1
                kl_loss = kl_div_estimate.mean()

                # --- Entropy loss ---
                entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-current_log_probs)

                # --- Total loss ---
                loss = policy_loss + self.kl_beta * kl_loss + self.ent_coef * entropy_loss

                # --- Logging ---
                pg_losses.append(policy_loss.item())
                kl_losses.append(kl_loss.item())
                entropy_losses.append(entropy_loss.item())
                losses.append(loss.item())
                clip_fraction = th.mean((th.abs(ratios - 1.0) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # --- Backprop ---
                self.policy.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1  # this is in accordance with PPO from SB3

        return pg_losses, kl_losses, entropy_losses, clip_fractions, losses

    def _train_outcome_supervision(self, clip_range: float) -> tuple[list, list, list, list, list]:
        """
        Outcome supervision training method.
        """
        pg_losses, kl_losses, entropy_losses, clip_fractions, losses = [], [], [], [], []

        advantages = self.group_rollout_buffer.get_advantages()  # shape: (n_trajectories, )

        for _ in range(self.n_epochs):

            obs_list, actions_list, old_log_probs_list, advantages_list = [], [], [], []

            for traj_idx, traj in enumerate(self.group_rollout_buffer.trajectories):
                obs, actions, old_log_probs = traj.to_tensor()
                old_log_probs = old_log_probs.detach()

                if len(obs) == 0:
                    continue

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = actions.long().flatten()

                current_log_probs, entropy = self.policy.evaluate_actions(obs, actions)

                advantage = th.as_tensor(advantages[traj_idx], dtype=th.float32, device=self.device)  # scalar tensor
                # TODO check if this tensor is correct

                if self.batch_group_updates:
                    # Only collect values for batched policy update for the whole group
                    obs_list.append(obs)
                    actions_list.append(actions)
                    old_log_probs_list.append(old_log_probs)
                    advantages_list.append(advantage)

                else:
                    # Policy update for each trajectory separately
                    current_log_probs_sum = current_log_probs.sum()  # sum of log probs = log of product of probs
                    old_log_probs_sum = old_log_probs.sum()

                    # --- Policy Gradient loss ---
                    ratio = th.exp(current_log_probs_sum - old_log_probs_sum)
                    surr1 = ratio * advantage
                    surr2 = th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage
                    policy_loss = -th.min(surr1, surr2)  # scalar tensor

                    # --- KL loss ---
                    with th.no_grad():
                        log_probs_ref, _ = self.policy_ref.evaluate_actions(obs, actions)
                        log_probs_ref_sum = log_probs_ref.sum()
                    kl_ratio = log_probs_ref_sum - current_log_probs_sum  # scalar tensor
                    kl_div_estimate = th.exp(kl_ratio) - kl_ratio - 1
                    kl_loss = kl_div_estimate

                    # --- Entropy loss ---
                    if entropy is None:
                        entropy_loss = current_log_probs_sum
                    else:
                        entropy_loss = -entropy.sum()

                    # --- Total loss ---
                    loss = policy_loss + self.kl_beta * kl_loss + self.ent_coef * entropy_loss

                    # --- Logging ---
                    pg_losses.append(policy_loss.item())
                    kl_losses.append(kl_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    losses.append(loss.item())
                    clip_fraction = float(abs(ratio.item() - 1.0) > clip_range)
                    clip_fractions.append(clip_fraction)

                    # --- Backprop ---
                    self.policy.optimizer.zero_grad()
                    loss.backward()  # computes gradients from the loss w.r.t policy parameters
                    if self.max_grad_norm is not None:
                        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

            if self.batch_group_updates:
                # Finally the batched policy update for the whole group
                obs = th.cat(obs_list, dim=0)
                actions = th.cat(actions_list, dim=0)
                old_log_probs = th.cat(old_log_probs_list, dim=0)
                advantages_tensor = th.cat(advantages_list, dim=0)

                current_log_probs, entropy = self.policy.evaluate_actions(obs, actions)

                # --- Policy Gradient loss ---
                ratios = th.exp(current_log_probs - old_log_probs)
                surr1 = ratios * advantages_tensor
                surr2 = th.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * advantages_tensor
                policy_loss = -th.min(surr1, surr2).mean()

                # --- KL loss ---
                with th.no_grad():
                    log_probs_ref, _ = self.policy_ref.evaluate_actions(obs, actions)
                kl_ratios = log_probs_ref - current_log_probs
                kl_div_estimate = th.exp(kl_ratios) - kl_ratios - 1
                kl_loss = kl_div_estimate.mean()

                # --- Entropy loss ---
                entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-current_log_probs)

                # --- Total loss ---
                loss = policy_loss + self.kl_beta * kl_loss + self.ent_coef * entropy_loss

                # --- Logging ---
                pg_losses.append(policy_loss.item())
                kl_losses.append(kl_loss.item())
                entropy_losses.append(entropy_loss.item())
                losses.append(loss.item())
                clip_fraction = th.mean((th.abs(ratios - 1.0) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # --- Backprop ---
                self.policy.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1

        return pg_losses, kl_losses, entropy_losses, clip_fractions, losses

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
    ) -> SelfGRPO:
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

            if self.policy_ref is None or iteration % self.kl_ref_iterations == 0:
                self.policy_ref = self.policy.get_frozen_deepcopy()  # set reference policy to the current policy
            self.train()  # Update the policy params based on the collected group rollouts

            iteration += 1

        return self
