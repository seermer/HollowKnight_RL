import random

import numpy as np


class SumTree:
    """
    an implementation of fixed size sumtree
    """

    def __init__(self, maxlen: int,
                 alpha: float = 0.6, beta: float = 0.4,
                 beta_anneal: float = 0.):
        self.maxlen = maxlen
        self.leaves = [None for _ in range(maxlen)]
        self.tree = np.zeros((maxlen * 2 - 1,), dtype=np.float64)
        self.alpha = alpha
        self.beta = beta
        self.beta_anneal = beta_anneal
        self.length = 0
        self.oldest = 0
        self.max_prio = 1.

    def _get_idx(self, val: float):
        idx = 0
        while True:
            left_idx = 2 * idx + 1
            if left_idx >= len(self.tree):
                break  # leaf

            if val <= self.tree[left_idx]:
                # traverse left
                idx = left_idx
            else:
                # traverse right
                val -= self.tree[left_idx]
                idx = left_idx + 1
        return idx + 1 - self.maxlen

    def update_prio(self, prio: float, idx: int, return_w: bool = True):
        if prio > self.max_prio:
            self.max_prio = prio
        prio = (prio + 1e-7) ** self.alpha

        idx = self.maxlen - 1 + idx
        old_prio = self.tree[idx]
        total = self.tree[0]
        gap = prio - old_prio

        while idx >= 0:
            self.tree[idx] += gap
            idx = (idx - 1) // 2
        if return_w:
            w = (1. / (self.length * old_prio / total)) ** self.beta
            return w

    def append(self, element):
        idx = self.oldest
        self.leaves[idx] = element
        self.update_prio(self.max_prio, idx, return_w=False)
        self.oldest = (idx + 1) % self.maxlen
        if self.length < self.maxlen:
            self.length += 1
        return idx

    def sample(self, k: int):
        assert k <= self.length
        segment = self.tree[0] / k
        indices = []
        elements = []
        for i in range(k):
            val = random.uniform(i * segment, (i + 1) * segment)
            idx = self._get_idx(val)
            while self.leaves[idx] is None:
                # failsafe when the tree returns an invalid index
                # because of float point error
                val = random.uniform(i * segment, (i + 1) * segment)
                idx = self._get_idx(val)
            indices.append(idx)
            elements.append(self.leaves[idx])
        return elements, indices

    def step_beta(self):
        self.beta += self.beta_anneal

    def __len__(self):
        return self.length

    def __str__(self):
        s = [f'SumTree:{self.length}']
        prev = 0
        idx = 1
        while True:
            s.append(str(self.tree[prev:idx]))
            if idx >= len(self.tree):
                break
            prev = idx
            idx = idx * 2 + 1

        s.append('Leaves')
        s.append(str(self.leaves))
        return '\n'.join(s)


def test():
    tree = SumTree(10)
    for i in range(4):
        idx = tree.append((i,))
        tree.update_prio(random.uniform(0, 1), idx)

    while True:
        tree.update_prio(random.uniform(0, 1), random.randint(0, 3))
        sample, indices = tree.sample(3)


if __name__ == '__main__':
    test()
