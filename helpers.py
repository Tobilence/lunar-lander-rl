from typing import Any, List
from random import sample

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
    
    def __len__(self):
        return self._get_buffer_size()
    
    def _get_buffer_size(self):
        # not the most efficient way, but works for now
        try:
            return self.buffer.index(None)
        except ValueError:
            return self.max_size