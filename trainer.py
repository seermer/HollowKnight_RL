import os
import copy
import time
import torch
import random
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    GAP = 0.15

    def __init__(self, env, replay_buffer,
                 n_frames, gamma, eps, eps_func, target_steps,
                 model, lr, criterion, batch_size, device,
                 is_double=True, frame_skip=True,
                 save_loc=None, no_save=False):
        self.env = env
        self.replay_buffer = replay_buffer

        assert n_frames > 0
        if frame_skip:
            self.n_frames = n_frames * 2 - 1
        else:
            self.n_frames = n_frames
        self.gamma = gamma
        self.eps = eps
        self.eps_func = eps_func
        self.target_steps = target_steps

        self.model = model.to(device)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-4)
        self.model.eval()
        self.target_model.eval()
        self.criterion = criterion
        if hasattr(self.criterion, 'to'):
            self.criterion = self.criterion.to(device)
        self.batch_size = batch_size
        self.device = device

        self.is_double = is_double
        self.frame_skip = frame_skip

        self.steps = 0
        self.episodes = 0
        self.target_replace_steps = 0

        self.no_save = no_save
        if not no_save:
            self.save_loc = ('./saved/' + str(int(time.time()))) if save_loc is None else save_loc
            if not self.save_loc.endswith('/'):
                self.save_loc += '/'
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

    def _process_frames(self, frames):
        if self.frame_skip:
            return tuple((frame for i, frame in enumerate(frames) if i % 2 == 0))
        else:
            return tuple(frames)

    @torch.no_grad()
    def _warmup(self, model):
        n_frames = (self.n_frames + 1) // 2 if self.frame_skip else self.n_frames
        model(torch.rand((1, n_frames) + self.env.observation_space.shape,
                         dtype=torch.float32, device=self.device))

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
        while True:
            t = time.time()
            obs_tuple = self._process_frames(stacked_obs)
            model_input = np.array([obs_tuple], dtype=np.float32)
            action = self.get_action(model_input)
            # note that we intentionally run the prediction no matter what epsilon is,
            # even when predicted action is immediately ignored,
            # so that fps can be slightly more stable
            if random_action or self.eps > random.uniform(0, 1):
                action = self.env.action_space.sample()
            obs_next, rew, done, _, _ = self.env.step(action)
            total_rewards += rew
            self.steps += 1
            stacked_obs.append(obs_next)
            obs_next_tuple = self._process_frames(stacked_obs)
            self.replay_buffer.add(obs_tuple, action, rew, obs_next_tuple, done)
            if not random_action:
                self.eps = self.eps_func(self.eps, self.episodes, self.steps)
                if len(self.replay_buffer) > self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)
                    self.learn(*batch)
            if done:
                break
            t = time.time() - t
            if t < self.GAP and not no_sleep:
                time.sleep(self.GAP - t)
        return total_rewards

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
            done = torch.as_tensor(done,
                                   dtype=torch.float32,
                                   device=self.device)
            obs_next = self._preprocess(obs_next)

            target_q = self.target_model(obs_next).detach()
            if self.is_double:
                self.model.eval()
                max_act = torch.argmax(self.model(obs_next).detach(), dim=1)
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
        obs = self._preprocess(obs)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        q = self.model(obs)
        q = torch.gather(q, 1, act)
        loss = self.criterion(q, target)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.target_replace_steps += 1
            loss = float(loss.detach().cpu().numpy())
            if self.target_replace_steps >= self.target_steps:
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_replace_steps = 0
                self.target_model.eval()
        return loss

    def save_models(self, prefix=''):
        if not self.no_save:
            torch.save(self.model.state_dict(), self.save_loc + prefix + 'model.pt')
            torch.save(self.target_model.state_dict(), self.save_loc + prefix + 'target_model.pt')
            torch.save(self.optimizer.state_dict(), self.save_loc + prefix + 'optimizer.pt')

    def log(self, info):
        if not self.no_save:
            for k, v in info.items():
                self.writer.add_scalar(k, v, self.episodes)
