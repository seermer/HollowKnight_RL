import os
import copy
import time
import torch
import random
import numpy as np
from collections import deque
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    GAP = 0.26

    def __init__(self, env, replay_buffer,
                 n_frames, gamma, eps, eps_func, target_steps, learn_freq,
                 model, lr, criterion, batch_size, device,
                 is_double=True, DrQ=True,
                 save_loc=None, no_save=False):
        self.env = env
        self.replay_buffer = replay_buffer

        assert n_frames > 0
        self.n_frames = n_frames
        self.gamma = gamma
        self.eps = eps
        self.eps_func = eps_func
        self.target_steps = target_steps
        self.learn_freq = learn_freq

        self.model = model.to(device)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1.5e-4)
        self.model.eval()
        self.target_model.eval()
        self.criterion = criterion
        if hasattr(self.criterion, 'to'):
            self.criterion = self.criterion.to(device)
        self.batch_size = batch_size
        self.device = device

        self.is_double = is_double
        self.DrQ = DrQ
        self.transform = T.RandomCrop(size=self.env.observation_space.shape,
                                      padding=8,
                                      padding_mode='edge')

        self.steps = 0
        self.episodes = 0
        self.target_replace_steps = 0

        self.no_save = no_save
        if not no_save:
            save_loc = ('./saved/' + str(int(time.time()))) if save_loc is None else save_loc
            assert not save_loc.endswith('\\')
            save_loc = save_loc if save_loc.endswith('/') else f'{save_loc}/'
            self.save_loc = save_loc
            if not os.path.exists(self.save_loc):
                os.makedirs(self.save_loc)
            self.writer = SummaryWriter(self.save_loc + 'log/')

        self._warmup(self.model)
        self._warmup(self.target_model)

    @staticmethod
    def _preprocess(obs):
        if len(obs.shape) > 3:  # image
            obs /= 127.5
            obs -= 1.
        return obs

    @staticmethod
    def _process_frames(frames):
        return tuple(frames)

    def augment(self, image):
        return [self.transform(image)
                for _ in range(2)]

    @torch.no_grad()
    def _warmup(self, model):
        model(torch.rand((1, self.n_frames) + self.env.observation_space.shape,
                         dtype=torch.float32,
                         device=self.device)).detach().cpu().numpy()

    @torch.no_grad()
    def get_action(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32,
                              device=self.device)
        self.model.eval()
        pred = self.model(obs).detach().cpu().numpy()[0]
        return np.argmax(pred)

    def run_episode(self, random_action=False, no_sleep=False):
        self.episodes += 1

        initial, _ = self.env.reset()
        stacked_obs = deque(
            (initial for _ in range(self.n_frames)),
            maxlen=self.n_frames
        )
        total_rewards = 0
        total_loss = 0
        learned_times = 0
        while True:
            t = time.time()
            obs_tuple = self._process_frames(stacked_obs)
            if random_action or self.eps > random.uniform(0, 1):
                action = self.env.action_space.sample()
            else:
                model_input = np.array([obs_tuple], dtype=np.float32)
                action = self.get_action(model_input)
            obs_next, rew, done, _, _ = self.env.step(action)
            total_rewards += rew
            self.steps += 1
            stacked_obs.append(obs_next)
            self.replay_buffer.add(obs_tuple, action, rew, done)
            if not random_action:
                self.eps = self.eps_func(self.eps, self.episodes, self.steps)
                if len(self.replay_buffer) > self.batch_size and self.steps % self.learn_freq == 0:
                    batch = self.replay_buffer.sample(self.batch_size)
                    total_loss += self.learn(*batch)
                    learned_times += 1
            if done:
                break
            t = self.GAP - (time.time() - t)
            if t > 0 and not no_sleep:
                time.sleep(t)
        total_loss = total_loss / learned_times if learned_times > 0 else 0
        return total_rewards, total_loss

    def run_episodes(self, n, **kwargs):
        for _ in range(n):
            self.run_episode(**kwargs)

    def learn(self, obs, act, rew, obs_next, done):  # update with a given batch
        with torch.no_grad():
            act = torch.as_tensor(act,
                                  dtype=torch.int64,
                                  device=self.device)
            rew = torch.as_tensor(rew,
                                  dtype=torch.float32,
                                  device=self.device)
            obs_next = torch.as_tensor(
                obs_next,
                dtype=torch.float32,
                device=self.device
            ).squeeze(1)
            obs_next = torch.vstack(self.augment(obs_next))
            done = torch.as_tensor(done,
                                   dtype=torch.float32,
                                   device=self.device)
            obs_next = self._preprocess(obs_next)

            target_q = self.target_model(obs_next).detach()
            target_q = target_q[:self.batch_size] + target_q[self.batch_size:]
            target_q /= 2.
            if self.is_double:
                self.model.eval()
                max_act = self.model(obs_next).detach()
                max_act = max_act[:self.batch_size] + max_act[self.batch_size:]
                max_act = torch.argmax(max_act, dim=1)
                max_target_q = target_q[torch.arange(self.batch_size), max_act]
                max_target_q = max_target_q.unsqueeze(-1)
            else:
                max_target_q, _ = target_q.max(dim=1, keepdims=True)
            target = rew + self.gamma * max_target_q * (1. - done)

        obs = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device
        ).squeeze(1)
        obs = torch.vstack(self.augment(obs))
        obs = self._preprocess(obs)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        q = self.model(obs)
        q = (q[:self.batch_size] + q[self.batch_size:]) / 2.
        q = torch.gather(q, 1, act)
        loss = self.criterion(q, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()

        with torch.no_grad():
            self.target_replace_steps += 1
            loss = float(loss.detach().cpu().numpy())
            if self.target_replace_steps >= self.target_steps:
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_replace_steps = 0
                self.target_model.eval()
        return loss

    def load_explorations(self, save_loc='./explorations/'):
        assert not save_loc.endswith('\\')
        save_loc = save_loc if save_loc.endswith('/') else f'{save_loc}/'
        for file in os.listdir(save_loc):
            if not file.endswith('.npz'):
                continue
            fname = save_loc + file
            print('loading', os.path.abspath(fname))
            arrs = np.load(fname)
            obs_lst = arrs['o']
            action_lst = arrs['a']
            rew_lst = arrs['r']
            done_lst = arrs['d']
            assert obs_lst[0].shape == self.env.observation_space.shape
            assert (len(action_lst) == len(rew_lst) ==
                    len(done_lst) == len(obs_lst) - 1)
            stacked_obs = deque(
                (obs_lst[0] for _ in range(self.n_frames)),
                maxlen=self.n_frames
            )
            for o, a, r, d in zip(obs_lst[1:], action_lst, rew_lst, done_lst):
                obs_tuple = self._process_frames(stacked_obs)
                stacked_obs.append(o)
                self.replay_buffer.add(obs_tuple, a, r, d)
        print('loading complete, with buffer length', len(self.replay_buffer))

    def save_explorations(self, n_episodes, save_loc='./explorations/'):
        assert not save_loc.endswith('\\')
        save_loc = save_loc if save_loc.endswith('/') else f'{save_loc}/'
        for i in range(n_episodes):
            fname = f'{save_loc}{i}.npz'
            if os.path.exists(fname):
                print(f'{os.path.abspath(fname)} already exists, skipping')
                continue
            obs, _ = self.env.reset()
            obs_lst = [obs]
            action_lst, rew_lst, done_lst = [], [], []
            while True:
                t = time.time()
                action = self.env.action_space.sample()
                # predict with model to simulate the time taken in real episode
                self._warmup(self.model)
                obs_next, rew, done, _, _ = self.env.step(action)
                obs_lst.append(obs_next)
                action_lst.append(action)
                rew_lst.append(rew)
                done_lst.append(done)
                t = time.time() - t
                if t < self.GAP:
                    time.sleep(self.GAP - t)
                if done:
                    break
            obs_lst = np.array(obs_lst, dtype=obs.dtype)
            if max(action_lst) < 256:
                action_lst = np.array(action_lst, dtype=np.uint8)
            else:
                action_lst = np.array(action_lst, dtype=np.uint64)
            rew_lst = np.array(rew_lst, dtype=np.float32)
            done_lst = np.array(done_lst, dtype=np.bool8)
            np.savez_compressed(fname, o=obs_lst, a=action_lst, r=rew_lst, d=done_lst)
            print(f'saved exploration at {os.path.abspath(fname)}')

    def save_models(self, prefix=''):
        if not self.no_save:
            torch.save(self.model.state_dict(), self.save_loc + prefix + 'model.pt')
            torch.save(self.target_model.state_dict(), self.save_loc + prefix + 'target_model.pt')
            torch.save(self.optimizer.state_dict(), self.save_loc + prefix + 'optimizer.pt')

    def log(self, info):
        if not self.no_save:
            for k, v in info.items():
                self.writer.add_scalar(k, v, self.episodes)
