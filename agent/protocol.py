
from typing import Protocol
import numpy as np


class Agent(Protocol):
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Determines the next action based on the current state.
        Returns an integer (the action index).
        """
        ...
