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
    m = models.AttentionExtractor(env.observation_space.shape, n_frames, device=DEVICE)
    m = models.SinglePathMLP(m, env.action_space.n, False)
    return m


@torch.no_grad()
def main():
    n_frames = 5
    env = hkenv.HKEnv((224, 224), w1=1., w2=1., w3=0.)
    m = get_model(env, n_frames)
    m.eval()
    m.load_state_dict(torch.load('saved/1671598991/bestmodel.pt'))
    m(torch.ones((1, n_frames) + env.observation_space.shape,
                 dtype=torch.float32, device=DEVICE))
    for i in range(5):
        initial, _ = env.reset()
        stacked_obs = deque(
            (initial for _ in range(n_frames * 2 - 1)),
            maxlen=n_frames
        )
        while True:
            t = time.time()
            obs_tuple = tuple(stacked_obs)
            if random.uniform(0, 1) < 0.05:
                action = env.action_space.sample()
            else:
                obs = np.array([obs_tuple], dtype=np.float32)
                obs = torch.as_tensor(obs, dtype=torch.float32,
                                      device=DEVICE)
                pred = m(obs).detach().cpu().numpy()[0]
                action = np.argmax(pred)
            obs_next, rew, done, _, _ = env.step(action)
            print(rew)
            stacked_obs.append(obs_next)
            if done:
                break
            t = 0.15 - (time.time() - t)
            if t > 0:
                time.sleep(t)


if __name__ == '__main__':
    main()
