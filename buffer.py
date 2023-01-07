import random
import numpy as np
from collections import deque

from sumtree import SumTree


class Buffer:
    def __init__(self, size: int,
                 prioritized=None,
                 *args, **kwargs):
        assert size > 0
        self.buffer = (deque(maxlen=size) if prioritized is None
                       else SumTree(maxlen=size, **prioritized))
        self.maxlen = size
        self.prioritized = prioritized is not None
        self._temp_buffer = []

    @staticmethod
    def _to_numpy(batch):
        obs, act, rew, obs_next, done = [], [], [], [], []
        for o, a, r, o_, d in batch:
            obs.append(np.concatenate(o))
            act.append(a)
            rew.append(r)
            obs_next.append(np.concatenate(o_))
            done.append(d)

        return (np.array(obs, copy=True, dtype=np.float32),
                np.array(act, copy=True, dtype=np.int64)[:, np.newaxis],
                np.array(rew, copy=True, dtype=np.float32)[:, np.newaxis],
                np.array(obs_next, copy=True, dtype=np.float32),
                np.array(done, copy=True, dtype=np.float32)[:, np.newaxis])

    @property
    def is_full(self):
        return len(self.buffer) == self.maxlen

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
        return self._to_numpy(batch)

    def prioritized_sample(self, batch_size):
        batch, indices = self.buffer.sample(batch_size)
        return self._to_numpy(batch), indices

    def update_priority(self, priorities, indices):
        weights = []
        for prio, idx in zip(priorities, indices):
            weights.append(self.buffer.update_prio(prio, idx))
        # print(self.buffer)
        weights = np.array(weights, dtype=np.float32)
        return weights / np.max(weights)

    def step(self):
        if self.prioritized:
            self.buffer.step_beta()

    def __len__(self):
        return len(self.buffer)

    def __str__(self):
        return '\n'.join(map(str, self.buffer))


class MultistepBuffer(Buffer):
    def __init__(self, size: int, n: int = 5, gamma: float = 0.9,
                 prioritized=None):
        super(MultistepBuffer, self).__init__(size, prioritized=prioritized)

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
