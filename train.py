import torch
import tianshou as ts
from torch.backends import cudnn

import hkenv
import models

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True


def get_model(env: hkenv.HKEnv, n_frames: int):
    m = models.SimpleExtractor(env.observation_space.shape, n_frames, device=DEVICE)
    m = models.SinglePathMLP(m, env.action_space.n)
    return m


def train(env, policy, collector, episodes):
    print('training started')
    collector.collect(n_episode=1, random=True)
    # b = collector.buffer.sample_indices(1)
    # print(b)
    eps = 0.85
    for i in range(1, episodes + 1):
        policy.set_eps(max(eps ** i, 0.05))
        result = collector.collect(n_episode=1)  # TODO: stack for prediction
        env.close()
        losses = policy.update(64, collector.buffer)
        print(f'episode {i}:'
              f'collected {result["n/st"]}, reward {result["rew"]}, loss {losses["loss"]}')


def main():
    n_frames = 16
    e = hkenv.HKEnv((160, 160))
    m = get_model(e, n_frames)  # TODO: run warmup
    e = ts.env.DummyVectorEnv([lambda: e])
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
    policy = ts.policy.DQNPolicy(
        model=m,
        optim=optimizer,
        discount_factor=0.98,
        estimation_step=5,
        target_update_freq=30000
    )
    buffer = ts.data.PrioritizedVectorReplayBuffer(total_size=100000,
                                                   buffer_num=1,
                                                   alpha=0.6,
                                                   beta=0.4,
                                                   stack_num=n_frames)
    collector = ts.data.Collector(
        policy=policy,
        env=e,
        buffer=buffer,
        exploration_noise=True
    )
    train(e, policy, collector, 100)


if __name__ == '__main__':
    main()
