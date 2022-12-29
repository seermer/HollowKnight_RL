import random
import numpy as np
from collections import deque


class Buffer:
    def __init__(self, size: int, *args, **kwargs):
        assert size > 0
        self.buffer = deque(maxlen=size)
        self.max_len = size
        self._temp_buffer = []

    @property
    def is_full(self):
        return len(self.buffer) == self.max_len

    def add(self, obs, act, rew, done):
        self._temp_buffer.append((obs, act, rew, done))
        if len(self._temp_buffer) == 2:
            obs_, act_, rew_, done_ = self._temp_buffer.pop(0)
            self.buffer.append((obs_, act_, rew_, obs, done_))
            if done:
                self.buffer.append((obs, act, rew, obs, done))
                self._temp_buffer = []

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, obs_next, done = [], [], [], [], []
        for o, a, r, o_, d in batch:
            obs.append(o)
            act.append(a)
            rew.append(r)
            obs_next.append(o_)
            done.append(d)

        return (np.array(obs, copy=True, dtype=np.float32),
                np.array(act, copy=True, dtype=np.int64)[:, np.newaxis],
                np.array(rew, copy=True, dtype=np.float32)[:, np.newaxis],
                np.array(obs_next, copy=True, dtype=np.float32),
                np.array(done, copy=True, dtype=bool)[:, np.newaxis])

    def __len__(self):
        return len(self.buffer)

    def __str__(self):
        return '\n'.join(map(str, self.buffer))


class MultistepBuffer(Buffer):
    def __init__(self, size: int, n: int = 5, gamma: float = 0.9):
        super(MultistepBuffer, self).__init__(size)

        self.n = n
        self.gamma = gamma

    def _add_nstep(self, obs_next, done):
        record = self._temp_buffer.pop(0)
        obs, act, rew, _ = record
        for i, rec in enumerate(self._temp_buffer, 1):
            rew += (self.gamma ** i) * rec[2]
        self.buffer.append((obs, act, rew, obs_next, done))

    def add(self, obs, act, rew, done):
        self._temp_buffer.append((obs, act, rew, done))
        if len(self._temp_buffer) > self.n:
            self._add_nstep(obs, done)
        if done:
            while len(self._temp_buffer) > 0:
                self._add_nstep(obs, done)
