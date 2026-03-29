from typing import Any, List
import numpy as np
from random import sample
import jax.numpy as jnp

MAX_PROB_INT = 999 # assign a high number to increase probability of item being sampled

class RingBuffer:
    def __init__(self, max_size=500) -> None:
        assert max_size > 0
        self.max_size: int = max_size
        self.next_pos = 0
        self.current_buffer_length = 0

        self.buffer: List[Any] = [None for _ in range(self.max_size)]
        self.tdes: List[int] = [0 for _ in range(self.max_size)]

    def add(self, item: Any):
        if self.next_pos < self.max_size:
            self.buffer[self.next_pos] = item
            self.tdes[self.next_pos] = MAX_PROB_INT
            self.next_pos += 1
            self.current_buffer_length += 1
            self.current_buffer_length = min(self.max_size, self.current_buffer_length)
        else:
            self.buffer[0] = item
            self.next_pos = 1
        
    def priority_sample(self, n, priorization_alpha=0.6, priorization_epsilon=1e-3):
        size = self.current_buffer_length
        tdes = np.abs(self.tdes[:size]) + priorization_epsilon
    
        # Alpha determines how much prioritization to use
        # alpha=0 is uniform, alpha=1 is full proportional prioritization
        p = tdes ** priorization_alpha
        probs = p / p.sum()

        indices = np.random.choice(size, n, replace=False, p=probs)
        return [self.buffer[i] for i in indices], list(indices)
    
    def store_tdes(self, sampled_indices, tdes):
        for idx, tde in zip(sampled_indices, tdes):
            self.tdes[idx] = tde
    
    def sample_jax(self, n:int):
        m_batch, m_indices = self.priority_sample(n)
        states, actions, rewards, next_states, dones = zip(*m_batch)
        return {
            "states": jnp.array(states),
            "action": jnp.array(actions),
            "reward": jnp.array(rewards),
            "next_state": jnp.array(next_states),
            "dones": jnp.array(dones, dtype=jnp.bool_)
        }, m_indices

    
    def __len__(self):
        return self.current_buffer_length


# calculates a linear decay for epsilon
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

