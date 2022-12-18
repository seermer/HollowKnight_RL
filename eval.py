import random
import time
import gym
import torch
import numpy as np
from collections import deque
from torch.backends import cudnn

import hkenv
import models

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True


def get_model(env: gym.Env, n_frames: int):
    m = models.SimpleExtractor(env.observation_space.shape, n_frames, device=DEVICE)
    m = models.DuelingMLP(m, env.action_space.n, False)
    return m


@torch.no_grad()
def main():
    n_frames = 5
    env = hkenv.HKEnv((160, 160), w1=1., w2=36., w3=0., no_magnitude=True)
    m = get_model(env, n_frames)
    m.eval()
    m.load_state_dict(torch.load('./saved/1671354397/bestmodel.pt'))
    m(torch.ones((1, n_frames) + env.observation_space.shape,
                 dtype=torch.float32, device=DEVICE))
    for i in range(5):
        initial, _ = env.reset()
        stacked_obs = deque(
            (initial for _ in range(n_frames)),
            maxlen=n_frames
        )
        while True:
            obs_tuple = tuple(stacked_obs)
            obs = np.array([obs_tuple], dtype=np.float32)
            obs = torch.as_tensor(obs, dtype=torch.float32,
                                  device=DEVICE)
            pred = m(obs).detach().cpu().numpy()[0]
            action = np.argmax(pred)
            obs_next, rew, done, _, _ = env.step(action)
            print(rew)
            time.sleep(0.042)
            stacked_obs.append(obs_next)
            if done:
                break


if __name__ == '__main__':
    main()
