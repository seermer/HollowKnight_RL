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
    m = models.ResidualExtractor(env.observation_space.shape, n_frames, device=DEVICE)
    m = models.DuelingMLP(m, env.action_space.n, False)
    return m


def train(dqn):
    print('training started')
    dqn.save_explorations(75)
    dqn.load_explorations()

    saved_rew = float('-inf')
    for i in range(1000):
        rew, loss = dqn.run_episode()
        if i % 10 == 0:
            eval_rew = dqn.evaluate()
            if eval_rew > saved_rew:
                saved_rew = eval_rew
                dqn.save_models('best')
        dqn.save_models('latest')

        dqn.log({'reward': rew, 'loss': loss})
        print(f'episode {dqn.episodes} finished, total step {dqn.steps}, epsilon {dqn.eps}',
              f'total rewards {rew}, loss {loss}', sep='\n')
        print()


def main():
    n_frames = 4
    env = hkenv.HKEnv((192, 192), w1=1., w2=1., w3=0.002)
    m = get_model(env, n_frames)
    replay_buffer = buffer.MultistepBuffer(80000, n=12, gamma=0.99)
    dqn = trainer.Trainer(env=env, replay_buffer=replay_buffer,
                          n_frames=n_frames, gamma=0.99, eps=1.,
                          eps_func=(lambda val, episode, step:
                                    max(0.1, val - 1e-5)),
                          target_steps=5000,
                          learn_freq=1,
                          model=m,
                          lr=1e-3,
                          criterion=torch.nn.MSELoss(),
                          batch_size=32,
                          device=DEVICE,
                          is_double=True,
                          DrQ=True,
                          no_save=False)
    train(dqn)


if __name__ == '__main__':
    main()
