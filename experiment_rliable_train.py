"""
Module containing the main entry point for training custom algorithms in RL Zoo 3.
This script registers custom algorithms and starts the training process.
It is designed to be run as a standalone script.
"""

from rl_zoo3.train import train
from rl_zoo3.utils import ALGOS

from sb3_contrib.grpo.grpo import GRPO
from sb3_contrib.rloo.rloo import RLOO

# Register custom algorithms in RL Zoo 3
ALGOS["process-grpo"] = GRPO
ALGOS["deepseek-process-grpo"] = GRPO
ALGOS["outcome-grpo"] = GRPO
ALGOS["rloo"] = RLOO
# TODO add other custom algorithms as needed (e.g., CISPO)

# Start training
train()
