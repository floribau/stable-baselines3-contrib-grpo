"""
Module containing the RLOO class
"""

from typing import TypeVar

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.type_aliases import MaybeCallback

from sb3_contrib.common.buffers import DeepSeekOutcomeGroupBuffer
from sb3_contrib.grpo.grpo import GRPO

SelfRLOO = TypeVar("SelfRLOO", bound="RLOO")

class RLOO(GRPO):
    """
    Reinforce Leave-One-Out (RLOO)

    Paper: https://openreview.net/pdf?id=r1lgTGL5DE

    Introduction to RLOO: https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo

    :param policy: The policy model to use (MlpPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param group_size: The number of trajectories to collect in a group.
    :param gamma: Discount factor. Usually, it is set to 1.
    :param scale_rewards: Whether to scale rewards by the standard deviation during advantage calculation.
    :param kl_beta: KL divergence penalty coefficient for the loss calculation.
    :param kl_ref_iterations: Number of learning iterations before updating the reference policy for KL divergence penalty.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
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

    def __init__(
        self,
        policy,
        env,
        learning_rate=0.0003,
        group_size=64,
        gamma=1,
        scale_rewards=False,
        kl_beta=0.01,
        kl_ref_iterations=1,
        ent_coef=0.005,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        group_rollout_buffer_class=DeepSeekOutcomeGroupBuffer,
        group_rollout_buffer_kwargs=None,
        stats_window_size=100,
        tensorboard_log=None,
        monitor_wrapper=True,
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            group_size=group_size,
            n_epochs=1,  # Only one epoch in RLOO because policy can't change in one training iteration
            gamma=gamma,
            clip_range=float("inf"),  # RLOO doesn't clip advantages
            batch_group_updates=True,  # RLOO is batched by definition
            scale_rewards=scale_rewards,
            kl_beta=kl_beta,
            kl_ref_iterations=kl_ref_iterations,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            group_rollout_buffer_class=group_rollout_buffer_class,
            group_rollout_buffer_kwargs=group_rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            monitor_wrapper=monitor_wrapper,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def train(self) -> None:
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        pg_losses, kl_losses, entropy_losses, losses = self._train()

        # Logs
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/kl_loss", np.mean(kl_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/total_loss", np.mean(losses))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def _train(self) -> tuple[list, list, list, list]:
        """
        RLOO training method (outcome supervision by definition).
        """
        # RLOO outcome supervision update method
        self.policy.set_training_mode(False)
        advantages = self.group_rollout_buffer.get_leave_one_out_advantages()  # shape: (n_trajectories, )
        grads, log_probs_sums, entropies = [], [], []

        for traj_idx, traj in enumerate(self.group_rollout_buffer.trajectories):
            # Only collect values for batched policy update for the whole group
            obs, actions, old_log_probs = traj.to_tensor()  # per step log probs
            advantage = th.as_tensor(advantages[traj_idx], dtype=th.float32, device=self.device)

            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            current_log_probs, entropy = self.policy.evaluate_actions(obs, actions)
            # Sanity check: current log probs equals old log probs because policy is the same
            assert th.allclose(
                current_log_probs, old_log_probs, rtol=1e-4, atol=1e-6
            ), f"Log probs {current_log_probs} should equal {old_log_probs}"
            log_probs_sum = current_log_probs.sum()  # sum of log probs = log of product of probs

            grads.append(log_probs_sum * advantage)
            log_probs_sums.append(log_probs_sum)
            entropies.append(entropy.sum())

        self.policy.set_training_mode(True)

        # --- Policy Gradient loss ---
        assert len(advantages) > 1
        pg_loss = -th.stack(grads).mean()

        # --- KL loss ---
        with th.no_grad():
            log_probs_ref, _ = self.policy_ref.evaluate_actions(obs, actions)
        kl_ratios = log_probs_ref - current_log_probs
        kl_div_estimate = th.exp(kl_ratios) - kl_ratios - 1
        kl_loss = kl_div_estimate.mean()

        # --- Entropy loss ---
        entropy_tensor = th.tensor(entropies)
        entropy_loss = -entropy_tensor.mean()

        # --- Total loss ---
        loss = pg_loss + self.kl_beta * kl_loss + self.ent_coef * entropy_loss

        # --- Backprop ---
        self.policy.optimizer.zero_grad()
        loss.backward()  # computes gradients from the loss w.r.t policy parameters
        if self.max_grad_norm is not None:
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        self._n_updates += 1

        # pg_losses, entropy_losses, total_losses
        return [pg_loss.item()], [kl_loss.item()], [entropy_loss.item()], [loss.item()]

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 10,
        tb_log_name: str = "ProcessSupervisionGRPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfRLOO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
