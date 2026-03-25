from typing import Any, List
from random import sample
import jax.numpy as jnp

class RingBuffer:
    def __init__(self, max_size=500) -> None:
        assert max_size > 0
        self.max_size: int = max_size
        self.buffer: List[Any] = [None for _ in range(self.max_size)]
        self.next_pos = 0

    def add(self, item: Any):
        if self.next_pos < self.max_size:
            self.buffer[self.next_pos] = item
            self.next_pos += 1
        else:
            self.next_pos = 0
            self.buffer[self.next_pos] = item
        
    def sample(self, n):
        size = self._get_buffer_size()
        return sample(self.buffer[:size], n)
    
    def sample_jax(self, n):
        m_batch = self.sample(n)
        states, actions, rewards, next_states, dones = zip(*m_batch)
        return {
            "states": jnp.array(states),
            "action": jnp.array(actions),
            "reward": jnp.array(rewards),
            "next_state": jnp.array(next_states),
            "dones": jnp.array(dones, dtype=jnp.bool_)
        }

    
    def __len__(self):
        return self._get_buffer_size()
    
    def _get_buffer_size(self):
        # not the most efficient way, but works for now
        try:
            return self.buffer.index(None)
        except ValueError:
            return self.max_size


def calculate_epsilon_decay(
        current_step,
        epsilon_start,
        epsilon_end,
        epsilon_decay_steps,
    ):
    return max(
        epsilon_end,
        epsilon_start - (epsilon_start - epsilon_end) * (current_step / epsilon_decay_steps)
    )

