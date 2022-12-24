import gym
import torch
import numpy as np
from torch import nn
from torch.backends import cudnn

import trainer
import buffer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True


class Net(nn.Module):
    def __init__(self, inp, out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(inp, 64)
        self.linear2 = nn.Linear(64, out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


def get_model(env: gym.Env):
    m = Net(np.prod(env.observation_space.shape), env.action_space.n)
    return m


def train(dqn):
    print('training started')
    dqn.run_episodes(n=100, random_action=True, no_sleep=True)

    for i in range(1000):
        dqn.run_episode(no_sleep=True)
        if i % 10 == 0:
            print('episode', i)
            dqn.evaluate(no_sleep=True)
            print()


def main():
    n_frames = 1
    env = gym.make('CartPole-v0')
    m = get_model(env)
    replay_buffer = buffer.MultistepBuffer(500000, n=12, gamma=0.99)
    dqn = trainer.Trainer(env=env, replay_buffer=replay_buffer,
                          n_frames=n_frames, gamma=0.99, eps=0.1,
                          eps_func=(lambda val, episode, step:
                                    0.1),
                          target_steps=2000,
                          learn_freq=1,
                          model=m,
                          lr=1e-3,
                          criterion=torch.nn.MSELoss(),
                          batch_size=32,
                          device=DEVICE,
                          is_double=True,
                          DrQ=False,
                          no_save=True)
    train(dqn)


if __name__ == '__main__':
    main()
