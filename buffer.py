import random
import numpy as np
from collections import deque


class Buffer:
    def __init__(self, size: int):
        assert size > 0
        self.buffer = deque(maxlen=size)

    def add(self, obs, act, rew, obs_next, done):
        self.buffer.append((obs, act, rew, obs_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, obs_next, done = [], [], [], [], []
        for o, a, r, o_, d in batch:
            o_ = o[1:] + (o_,)  # frame stack next obs
            obs.append(o)
            act.append(a)
            rew.append(r)
            obs_next.append(o_)
            done.append(d)

        # avoid copy if possible (usually have to copy though)
        return (np.array(obs, copy=False),
                np.array(act, copy=False, dtype=np.int64),
                np.array(rew, copy=False, dtype=np.float32),
                np.array(obs_next, copy=False),
                np.array(done, copy=False, dtype=bool))

    def __len__(self):
        return len(self.buffer)
