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
    m = models.SimpleExtractor(env.observation_space.shape, n_frames)
    m = models.DuelingMLP(m, env.action_space.n, noisy=True)
    return m.to(DEVICE)


def train(dqn):
    print('training started')
    dqn.save_explorations(75)
    dqn.load_explorations()
    # raise ValueError
    dqn.learn()  # warmup

    saved_rew = float('-inf')
    saved_train_rew = float('-inf')
    for i in range(1, 501):
        print('episode', i)
        rew, loss, lr = dqn.run_episode()
        if rew > saved_train_rew:
            print('new best train model found')
            saved_train_rew = rew
            dqn.save_models('besttrain')
        if i % 10 == 0:
            dqn.run_episode(random_action=True)
            eval_rew = dqn.evaluate()
            if eval_rew > saved_rew:
                print('new best eval model found')
                saved_rew = eval_rew
                dqn.save_models('best')
        dqn.save_models('latest')

        dqn.log({'reward': rew, 'loss': loss}, i)
        print(f'episode {i} finished, total step {dqn.steps}, learned {dqn.learn_steps}, epsilon {dqn.eps}',
              f'total rewards {round(rew, 3)}, loss {round(loss, 3)}, current lr {round(lr, 8)}', sep='\n')
        print()


def main():
    n_frames = 4
    env = hkenv.HKEnv((160, 160), w1=0.8, w2=0.8, w3=-0.0001)
    m = get_model(env, n_frames)
    replay_buffer = buffer.MultistepBuffer(100000, n=20, gamma=0.99)
    dqn = trainer.Trainer(env=env, replay_buffer=replay_buffer,
                          n_frames=n_frames, gamma=0.99, eps=0.,
                          eps_func=(lambda val, step: 0.),
                          target_steps=6000,
                          learn_freq=0.5,
                          model=m,
                          lr=1e-4,
                          criterion=torch.nn.MSELoss(),
                          batch_size=32,
                          device=DEVICE,
                          is_double=True,
                          DrQ=True,
                          reset=20000,
                          no_save=False)
    train(dqn)


if __name__ == '__main__':
    main()
