import gym
import torch
from torch.backends import cudnn

import hkenv
import models
import trainer
import buffer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True


def get_model(env: gym.Env, n_frames: int):
    m = models.SimpleExtractor(env.observation_space.shape, n_frames, device=DEVICE)
    m = models.DuelingMLP(m, env.action_space.n, False)
    return m


def train(dqn):
    print('training started')
    dqn.run_episodes(6, random_action=True)

    saved_rew = float('-inf')
    for i in range(3000):
        rew = dqn.run_episode()
        if rew >= saved_rew:
            saved_rew = rew
            dqn.save_models()
        dqn.save_models('latest')
        loss = 0
        n_batches = 32
        worst = None
        highest_loss = float('-inf')
        for _ in range(n_batches):
            batch = dqn.replay_buffer.sample(dqn.batch_size)
            cur_loss = dqn.learn(*batch)
            loss += cur_loss
            if cur_loss > highest_loss:
                highest_loss = cur_loss
                worst = batch
        n_batches += 1
        loss += dqn.learn(*worst)

        dqn.log({'reward': rew, 'loss': loss})
        print(f'episode {dqn.episodes} finished, total step {dqn.steps}, epsilon {dqn.eps}',
              f'total rewards {rew}, loss {loss / n_batches}', sep='\n')
        print()


def main():
    n_frames = 10
    env = hkenv.HKEnv((160, 160), w1=1., w2=36., w3=0., no_magnitude=True)
    m = get_model(env, n_frames)
    replay_buffer = buffer.Buffer(50000)
    dqn = trainer.Trainer(env=env, replay_buffer=replay_buffer,
                          n_frames=n_frames, gamma=0.99, eps=1.,
                          eps_func=(lambda val, episode, step:
                                    max(0.1, val - 1e-5)),
                          target_steps=12000,
                          model=m,
                          lr=2e-4,
                          criterion=torch.nn.HuberLoss(),
                          batch_size=32,
                          device=DEVICE,
                          is_double=True)
    train(dqn)


if __name__ == '__main__':
    main()
