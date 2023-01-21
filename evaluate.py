import gym
import torch
from torch.backends import cudnn

import hkenv
import models
import trainer
import buffer

DEVICE = 'cuda'
cudnn.benchmark = True


def get_model(env: gym.Env, n_frames: int):
    c, *shape = env.observation_space.shape
    m = models.SimpleExtractor(shape, n_frames * c)
    m = models.DuelingMLP(m, env.action_space.n, noisy=True)
    m = m.to(DEVICE)
    # modify below path to the weight file you have
    m.load_state_dict(torch.load('saved/1673754862HornetPER/bestmodel.pt'))
    return m


def evaluate(dqn):
    for _ in range(5):
        rew = dqn.evaluate()


def main():
    n_frames = 4
    env = hkenv.HKEnv((160, 160), rgb=False, gap=0.165, w1=1, w2=1, w3=0)
    m = get_model(env, n_frames)
    replay_buffer = buffer.MultistepBuffer(100000, n=10, gamma=0.99)
    dqn = trainer.Trainer(env=env, replay_buffer=replay_buffer,
                          n_frames=n_frames, gamma=0.99, eps=0.,
                          eps_func=(lambda val, step: 0.),
                          target_steps=6000,
                          learn_freq=1,
                          model=m,
                          lr=9e-5,
                          lr_decay=False,
                          criterion=torch.nn.MSELoss(),
                          batch_size=32,
                          device=DEVICE,
                          is_double=True,
                          drq=True,
                          svea=True,
                          reset=0,
                          no_save=True)
    evaluate(dqn)


if __name__ == '__main__':
    main()
