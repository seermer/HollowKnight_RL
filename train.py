import gym
import torch
import psutil
from torch.backends import cudnn

import hkenv
import models
import trainer
import buffer

DEVICE = 'cuda'
cudnn.benchmark = True


def get_model(env: gym.Env, n_frames: int):
    c, *shape = env.observation_space.shape
    m = models.SimpleExtractor(shape, n_frames * c,
                               activation='relu', sn=False)
    m = models.DuelingMLP(m, env.action_space.n,
                          activation='relu', noisy=True, sn=False)
    return m.to(DEVICE)


def train(dqn):
    print('training started')
    dqn.save_explorations(75)
    dqn.load_explorations()
    # dqn.load_explorations('saved/1673839254HornetTweaks/transitions')
    # raise ValueError
    dqn.learn()  # warmup

    saved_rew = float('-inf')
    saved_train_rew = float('-inf')
    for i in range(1, 550):
        print('episode', i)
        rew, loss, lr = dqn.run_episode()
        if rew > saved_train_rew and dqn.eps < 0.11:
            print('new best train model found')
            saved_train_rew = rew
            dqn.save_models('besttrain', online_only=True)
        if i % 10 == 0:
            dqn.run_episode(random_action=True)

            if i >= 100:
                eval_rew = dqn.evaluate()

                if eval_rew > saved_rew:
                    print('new best eval model found')
                    saved_rew = eval_rew
                    dqn.save_models('best', online_only=True)
        dqn.save_models('latest', online_only=True)

        dqn.log({'reward': rew, 'loss': loss, 'total steps': dqn.steps}, i)
        print(f'episode {i} finished, total step {dqn.steps}, learned {dqn.learn_steps}, epsilon {dqn.eps}',
              f'total rewards {round(rew, 3)}, loss {round(loss, 3)}, current lr {round(lr, 8)}',
              f'total memory usage {psutil.virtual_memory().percent}%', sep='\n')
        print()
    dqn.save_models('latest', online_only=False)


def main():
    n_frames = 4
    # env = hkenv.HKEnv((160, 160), rgb=False, gap=0.165, w1=0.8, w2=0.8, w3=-0.0001)
    env = hkenv.HKEnvV2((192, 192), rgb=False, gap=0.17, w1=0.8, w2=0.5, w3=-8e-5)
    m = get_model(env, n_frames)
    replay_buffer = buffer.MultistepBuffer(180000, n=10, gamma=0.99, prioritized=None)
                                           # prioritized={
                                           #     'alpha': 0.5,
                                           #     'beta': 0.4,
                                           #     'beta_anneal': 0.6 / 550.
                                           # })
    dqn = trainer.Trainer(env=env, replay_buffer=replay_buffer,
                          n_frames=n_frames, gamma=0.99, eps=0.,
                          eps_func=(lambda val, step: 0.),
                          target_steps=8000,
                          learn_freq=4,
                          model=m,
                          lr=8e-5,
                          lr_decay=False,
                          criterion=torch.nn.MSELoss(),
                          batch_size=32,
                          device=DEVICE,
                          is_double=True,
                          drq=True,
                          svea=True,
                          reset=0,  # no reset
                          n_targets=1,
                          save_suffix='HornetV2',
                          no_save=False)
    train(dqn)


if __name__ == '__main__':
    main()
