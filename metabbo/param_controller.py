"""Parameter controller interface.

Defines `ParamController` abstract interface that DQN-based controllers should implement.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

# Import global configuration
import sys
sys.path.insert(0, '/hada')
from config import set_global_seed, SEED

# Set global seed at module load time
set_global_seed()


class ParamController(ABC):
    @abstractmethod
    def reset(self, problem_meta: Dict[str, Any]) -> None:
        """Called at the start of optimization for a new problem/run."""

    @abstractmethod
    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        """Given an observation, return a dict of DE parameters to use."""