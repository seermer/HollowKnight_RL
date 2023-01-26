import gym
import torch
import numpy as np
from torch import nn
from torch.backends import cudnn

import trainer
import buffer
import models

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True


class DummyExtractor(nn.Module):
    def __init__(self, out_shape):
        super(DummyExtractor, self).__init__()
        self.out_shape = np.array([1, out_shape, 1])

    def forward(self, x):
        return x


class Net(models.AbstractFullyConnected):
    def __init__(self, inp, out):
        super(Net, self).__init__(DummyExtractor(out), out, noisy=True)
        self.linear1 = self.linear_cls(inp, 64)
        self.linear2 = self.linear_cls(64, out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, **kwargs):
        # x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


def get_model(env: gym.Env):
    m = Net(np.prod(env.observation_space.shape), env.action_space.n).to('cuda')
    return m


def train(dqn):
    print('training started')
    dqn.run_episodes(n=100, random_action=True)

    for i in range(1000):
        dqn.run_episode()
        if i % 10 == 0:
            print('episode', i)
            dqn.evaluate()
            print()


def main():
    n_frames = 1
    env = gym.make('CartPole-v0')
    m = get_model(env)
    replay_buffer = buffer.MultistepBuffer(500000, n=12, gamma=0.99,
                                           prioritized={
                                               'alpha': 0.6,
                                               'beta': 0.4,
                                               'beta_anneal': 0.6 / 300
                                           })
    dqn = trainer.Trainer(env=env, replay_buffer=replay_buffer,
                          n_frames=n_frames, gamma=0.99, eps=0.,
                          eps_func=(lambda val, step:
                                    0.),
                          target_steps=2000,
                          learn_freq=1,
                          model=m,
                          lr=1e-3,
                          lr_decay=False,
                          criterion=torch.nn.MSELoss(),
                          batch_size=32,
                          device=DEVICE,
                          is_double=True,
                          drq=False,
                          svea=False,
                          no_save=True)
    train(dqn)


if __name__ == '__main__':
    main()
