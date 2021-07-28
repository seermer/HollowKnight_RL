import numpy as np
import math
import random
from collections import deque


class ReplayBuffer:
    """
    this is NOT a priority experience replay buffer, but added some tricks to help training
    """

    def __init__(self, maxlen=4000):
        self.buffer1 = deque(maxlen=maxlen // 2)
        self.buffer2 = deque(maxlen=math.ceil(maxlen / 2))
        self.curr_len = 0
        self.median = 0

    def sample(self, n: int):
        if self.curr_len <= n:
            raise ValueError("no enough samples")
        priority_n = math.ceil(n * .65)
        priority_sample = random.sample(self.buffer1, k=priority_n)
        other_sample = random.sample(self.buffer2, k=n - priority_n)

        s, a, r, s_, d = [], [], [], [], []

        for sample in priority_sample:
            s.append(np.array(sample[0][:4]))
            s_.append(np.array(sample[0][1:]))
            a.append(sample[1])
            r.append(sample[2])
            d.append(sample[3])

        for sample in other_sample:
            s.append(np.array(sample[0][:4]))
            s_.append(np.array(sample[0][1:]))
            a.append(sample[1])
            r.append(sample[2])
            d.append(sample[3])
        return s, a, r, s_, d

    def append(self, states, action, reward, done):
        """
        note that states is in fact a combination of current state (s) and next state (s'),
        since there are overlaps between two states, and combining them will save some memory
        """
        if abs(reward) > .5:
            self.buffer1.append((states, action, reward, done))
        else:
            self.buffer2.append((states, action, reward, done))
