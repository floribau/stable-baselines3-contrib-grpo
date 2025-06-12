"""
TODO module doc
"""

import numpy as np
import torch as th
from gymnasium import spaces

from sb3_contrib.grpo.buffers import OutcomeGroupBuffer
from sb3_contrib.grpo.grpo import GRPO


class RLOO(GRPO):
    """
    TODO doc
    """

    def __init__(
        self,
        policy,
        env,
        learning_rate=0.0003,
        group_size=64,
        gamma=1,
        scale_rewards=False,
        ent_coef=0.005,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        group_rollout_buffer_class=OutcomeGroupBuffer,
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
            n_epochs=1,
            gamma=gamma,
            clip_range=float("inf"),
            batch_group_updates=True,
            use_importance_sampling=False,
            scale_rewards=scale_rewards,
            kl_beta=0,
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
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)

        pg_losses, entropy_losses, losses = self._train_rloo()

        # Logs
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/total_loss", np.mean(losses))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

    def _train_rloo(self) -> tuple[list, list, list]:
        """
        TODO doc
        NOTE this is always outcome supervision (is it?)
        """
        self.policy.set_training_mode(False)
        advantages = self.group_rollout_buffer.get_leave_one_out_advantages()  # shape: (n_trajectories, )
        grads, log_probs_sums, entropies = [], [], []

        for traj_idx, traj in enumerate(self.group_rollout_buffer.trajectories):
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

        # Policy gradient loss
        assert len(advantages) > 1
        pg_loss = -th.stack(grads).mean()

        # Entropy loss
        entropy_tensor = th.tensor(entropies)
        entropy_loss = -entropy_tensor.mean()

        loss = pg_loss + self.ent_coef * entropy_loss

        self.policy.optimizer.zero_grad()
        loss.backward()  # computes gradients from the loss w.r.t policy parameters
        if self.max_grad_norm is not None:
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        self._n_updates += 1

        # pg_losses, entropy_losses, total_losses
        return [pg_loss.item()], [entropy_loss.item()], [loss.item()]
