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
    dqn.save_explorations(50)
    dqn.load_explorations()

    saved_rew = float('-inf')
    for i in range(3000):
        rew, loss = dqn.run_episode()
        if rew >= saved_rew and dqn.eps < 0.1001:
            saved_rew = rew
            dqn.save_models('best')
        dqn.save_models('latest')

        dqn.log({'reward': rew, 'loss': loss})
        print(f'episode {dqn.episodes} finished, total step {dqn.steps}, epsilon {dqn.eps}',
              f'total rewards {rew}, loss {loss}', sep='\n')
        print()


def main():
    n_frames = 5
    env = hkenv.HKEnv((224, 224), w1=1., w2=1., w3=0.005)
    m = get_model(env, n_frames)
    replay_buffer = buffer.MultistepBuffer(30000)
    dqn = trainer.Trainer(env=env, replay_buffer=replay_buffer,
                          n_frames=n_frames, gamma=0.9, eps=1.,
                          eps_func=(lambda val, episode, step:
                                    max(0.1, val - 1e-5)),
                          target_steps=1600,
                          model=m,
                          lr=1e-4,
                          criterion=torch.nn.HuberLoss(),
                          batch_size=32,
                          device=DEVICE,
                          is_double=True,
                          no_save=False)
    train(dqn)


if __name__ == '__main__':
    main()
