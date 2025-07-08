import io
import pathlib
from typing import Any, TypeVar

import torch as th
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.ppo import PPO

SelfResetVfPPO = TypeVar("SelfResetVfPPO", bound="ResetVfPPO")


class ResetVfPPO(PPO):
    """
    ResetVfPPO is a variant of PPO that resets the value function parameters
    when loading a model. This is useful for scenarios where the value function
    needs to be reinitialized, such as when comparing to GRPO from a pretrained model.
    It inherits from PPO and overrides the `load` method to reset the value function parameters.
    """

    @classmethod
    def load(  # noqa: C901
        cls: type[SelfResetVfPPO],
        path: str | pathlib.Path | io.BufferedIOBase,
        env: GymEnv | None = None,
        device: th.device | str = "auto",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfResetVfPPO:
        model = super().load(
            path=path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs,
        )

        if hasattr(model.policy, "value_net"):
            model.policy.value_net.reset_parameters()

        return model
