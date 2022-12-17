import torch
import numpy as np
from torch.backends import cudnn

import hkenv
import models
import trainer
import buffer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True


def get_model(env: hkenv.HKEnv, n_frames: int):
    m = models.SimpleExtractor(env.observation_space.shape, n_frames, device=DEVICE)
    m = models.SinglePathMLP(m, env.action_space.n, False)
    return m


def train(dqn):
    print('training started')
    dqn.run_episodes(1, random_action=True)
    saved_rew = float('-inf')
    for i in range(300):
        rew = dqn.run_episode()
        if rew > saved_rew:
            saved_rew = rew
            dqn.save_models()

        loss = 0
        n_batches = 16
        best = None
        best_rew = float('-inf')
        for _ in range(n_batches):
            batch = dqn.replay_buffer.sample(32)
            batch_rew = np.max(batch[2])
            if batch_rew > best_rew:
                best_rew = batch_rew
                best = batch
            loss += dqn.learn(*batch)
        loss += dqn.learn(*best)

        dqn.log({'reward': rew, 'loss': loss})
        print(f'episode {dqn.episodes} finished, total step {dqn.steps}, epsilon {dqn.eps}',
              f'total rewards {rew}, loss {loss / (n_batches + 1)}', sep='\n')
        print()


def main():
    n_frames = 12
    env = hkenv.HKEnv((160, 160), w1=.95, w2=30., w3=0.)
    m = get_model(env, n_frames)
    replay_buffer = buffer.Buffer(50000)
    dqn = trainer.Trainer(env=env, replay_buffer=replay_buffer,
                          n_frames=n_frames, gamma=0.99, eps=1.,
                          eps_func=(lambda val, episode, step:
                                    max(0.1, val - 1e-4)),
                          target_steps=10000,
                          model=m,
                          lr=1e-3,
                          criterion=torch.nn.HuberLoss(),
                          batch_size=32,
                          device=DEVICE)
    train(dqn)


if __name__ == '__main__':
    main()
