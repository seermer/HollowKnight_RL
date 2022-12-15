import torch
from torch.backends import cudnn

import hkenv
import models
import trainer
import buffer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True


def get_model(env: hkenv.HKEnv, n_frames: int):
    m = models.SimpleExtractor(env.observation_space.shape, n_frames, device=DEVICE)
    m = models.SinglePathMLP(m, env.action_space.n)
    return m


def train(dqn):
    print('training started')
    dqn.run_episodes(1, random_action=True)

    for i in range(300):
        prev_step = dqn.steps
        rew = dqn.run_episode()
        loss = 0
        n_batches = (dqn.steps - prev_step) // 2
        for _ in range(n_batches):
            batch = dqn.replay_buffer.sample(32)
            loss += dqn.learn(*batch)
        dqn.log({'reward': rew, 'loss': loss})
        if i % 4 == 0:
            dqn.save_models()
        print(f'episode {dqn.episodes} finished, total step {dqn.steps},',
              f'total rewards {rew}, loss {loss / n_batches}', sep='\n')
        print()


def main():
    n_frames = 12
    env = hkenv.HKEnv((160, 160), w1=.95, w2=22., w3=0.08)
    m = get_model(env, n_frames)
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
    replay_buffer = buffer.Buffer(100000)
    dqn = trainer.Trainer(env=env, replay_buffer=replay_buffer,
                          n_frames=n_frames, gamma=0.98, eps=1.,
                          eps_func=(lambda val, episode, step:
                                    max(0.1, val - 9e-6)),
                          target_steps=10000,
                          model=m,
                          optimizer=optimizer,
                          criterion=torch.nn.MSELoss(),
                          device=DEVICE)
    train(dqn)


if __name__ == '__main__':
    main()
