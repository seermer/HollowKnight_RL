import copy
import torch
import random
import numpy as np
from collections import deque


class Trainer:
    def __init__(self, env, replay_buffer,
                 n_frames, gamma, eps, eps_func, target_steps,
                 model, optimizer, criterion, device):
        self.env = env
        self.replay_buffer = replay_buffer

        self.n_frames = n_frames
        self.gamma = gamma
        self.eps = eps
        self.eps_func = eps_func
        self.target_steps = target_steps

        self.model = model.to(device)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optimizer
        self.model.eval()
        self.target_model.eval()
        self.criterion = criterion
        if hasattr(self.criterion, 'to'):
            self.criterion = self.criterion.to(device)
        self.device = device
        self._warmup(self.model)
        self._warmup(self.target_model)

        self.steps = 0
        self.episodes = 0
        self.target_replace_steps = 0

    @staticmethod
    def _preprocess(obs):
        obs = obs / 127.5
        obs -= 1.
        return obs

    @torch.no_grad()
    def _warmup(self, model):
        model(torch.rand((1, self.n_frames) + self.env.observation_space.shape,
                         dtype=torch.float32, device=self.device))

    def get_action(self, obs):
        self.model.eval()
        pred = self.model(obs).detach().cpu().numpy()[0]
        return np.argmax(pred)

    def run_episode(self, random_action=False):
        self.episodes += 1
        stacked_obs = deque(
            (np.zeros(self.env.observation_space.shape, dtype=np.uint8)
             for _ in range(self.n_frames - 1)),
            maxlen=self.n_frames
        )
        obs, _ = self.env.reset()
        total_rewards = 0
        while True:
            stacked_obs.append(obs)
            model_input = np.array([stacked_obs], dtype=np.float32)
            with torch.no_grad():
                model_input = torch.as_tensor(model_input, dtype=torch.float32,
                                              device=self.device)
                action = self.get_action(model_input)
            # note that we intentionally run the policy no matter what epsilon is,
            # even when action is immediately ignored,
            # so that fps can be slightly more stable
            if random_action or self.eps > random.uniform(0, 1):
                action = self.env.action_space.sample()
            obs_next, rew, done, _, _ = self.env.step(action)
            total_rewards += rew
            self.steps += 1
            self.replay_buffer.add(tuple(stacked_obs), action, rew, obs_next, done)
            obs = obs_next
            if not random_action:
                self.eps = self.eps_func(self.eps, self.episodes, self.steps)
            if self.steps > 1000:
                batch = self.replay_buffer.sample(64)
                self.learn(*batch)
            if done:
                break
        self.env.cleanup()
        return total_rewards

    def run_episodes(self, n, random_action=False):
        for _ in range(n):
            self.run_episode(random_action=random_action)

    def learn(self, obs, act, rew, obs_next, done):  # update with a given batch
        self.target_replace_steps += len(done)
        obs = self._preprocess(obs)
        obs_next = self._preprocess(obs_next)
        with torch.no_grad():
            obs_next = torch.as_tensor(obs_next, dtype=torch.float32,
                                       device=self.device)
            target_q = self.target_model(obs_next).detach().cpu().numpy()
            max_target_q = target_q.max(axis=1)
            target = rew + self.gamma * max_target_q * ~done
            target = target[:, np.newaxis]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        obs = torch.as_tensor(obs, dtype=torch.float32,
                              device=self.device)
        target = torch.as_tensor(target, dtype=torch.float32,
                                 device=self.device)
        act = torch.as_tensor(act[:, np.newaxis], dtype=torch.int64,
                              device=self.device)
        q = self.model(obs)
        q = torch.gather(q, 1, act)
        loss = self.criterion(q, target)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            loss = float(loss.detach().cpu().numpy())
            if self.target_replace_steps >= self.target_steps:
                self.target_model.load_state_dict(self.model.state_dict())
                self.target_replace_steps = 0
        return loss
