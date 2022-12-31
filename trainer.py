import os
import copy
import time
import torch
import random
import numpy as np
from collections import deque
from kornia import augmentation as K
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    GAP = 0.16
    DEFAULT_STATS = (92.54949702814011,
                     57.94090462506912)

    def __init__(self, env, replay_buffer,
                 n_frames, gamma, eps, eps_func, target_steps, learn_freq,
                 model, lr, criterion, batch_size, device,
                 is_double=True, DrQ=True, reset=0,
                 save_loc=None, no_save=False):
        self.env = env
        self.replay_buffer = replay_buffer

        assert n_frames > 0
        self.n_frames = n_frames
        self.gamma = gamma
        self.eps = eps
        self.eps_func = eps_func
        self.target_steps = target_steps
        self.learn_freq = max(1, int(learn_freq))
        self.num_batches = max(1, int(1. / learn_freq))

        self.model = model.to(device)
        self.target_model = copy.deepcopy(self.model)
        self.init_lr = lr
        self.optimizer = torch.optim.NAdam(self.model.parameters(), lr=lr, eps=1.5e-4)
        self.model.eval()
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.criterion = criterion
        if hasattr(self.criterion, 'to'):
            self.criterion = self.criterion.to(device)
        self.batch_size = batch_size
        assert device == 'cuda', 'this version of code only supports cuda, ' \
                                 'so that mixed precision GradScaler can be used'
        self.device = device
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()

        self.is_double = is_double
        self.transform = K.RandomCrop(size=self.env.observation_space.shape,
                                      padding=(8, 8),
                                      padding_mode='replicate').to(device) if DrQ else None
        self.reset = reset
        self.stats = list(self.DEFAULT_STATS)

        self.steps = 0
        self.learn_steps = 0
        self.target_replace_times = 0
        self._learn_since_replace = 0

        self.no_save = no_save
        if not no_save:
            save_loc = ('./saved/'
                        + str(int(time.time()))
                        + 'Hornet2') if save_loc is None else save_loc
            assert not save_loc.endswith('\\')
            save_loc = save_loc if save_loc.endswith('/') else f'{save_loc}/'
            print('save files at', save_loc)
            self.save_loc = save_loc
            if not os.path.exists(self.save_loc):
                os.makedirs(self.save_loc)
            self.writer = SummaryWriter(self.save_loc + 'log/')

        self._warmup(self.model)
        self._warmup(self.target_model)

    @staticmethod
    def _process_frames(frames):
        return tuple(frames)

    def _standardize(self, obs):
        # values found from empirical data
        obs -= self.stats[0]
        obs /= self.stats[1]
        return obs

    def _preprocess(self, obs):
        if len(obs.shape) < 4:  # not image
            return torch.as_tensor(obs, dtype=torch.float32,
                                   device=self.device)
        obs = torch.as_tensor(obs, dtype=torch.float32,
                              device=self.device)
        self._standardize(obs)
        if self.transform:
            scale = torch.randn((self.batch_size, 1, 1, 1),
                                dtype=torch.float32, device=self.device)
            scale = torch.clip(scale, -2, 2) * 0.03 + 1.
            return torch.vstack((obs * scale, self.transform(obs)))
        else:
            return obs

    def _update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self._learn_since_replace = 0

    @torch.no_grad()
    def _warmup(self, model):
        model(torch.rand((1, self.n_frames) + self.env.observation_space.shape,
                         dtype=torch.float32,
                         device=self.device)).detach().cpu().numpy()

    @torch.no_grad()
    def get_action(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32,
                              device=self.device)
        if len(obs.shape) >= 4:
            self._standardize(obs)
        pred = self.model(obs, adv_only=True).detach().cpu().numpy()[0]
        return np.argmax(pred)

    def run_episode(self, random_action=False, no_sleep=False):
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
            if self.reset and self.steps % self.reset == 0:
                print('model reset')
                self.model.reset_linear()
                self._update_target()
            if not random_action:
                self.eps = self.eps_func(self.eps, self.steps)
                if len(self.replay_buffer) > self.batch_size and self.steps % self.learn_freq == 0:
                    # print(self.num_batches)
                    for _ in range(self.num_batches):
                        total_loss += self.learn()
                        learned_times += 1
            if done:
                break
            t = self.GAP - (time.time() - t)
            if t > 0 and not no_sleep:
                time.sleep(t)
            # print(t)
        total_loss = total_loss / learned_times if learned_times > 0 else 0
        return total_rewards, total_loss, self.optimizer.param_groups[0]['lr']

    def run_episodes(self, n, **kwargs):
        for _ in range(n):
            self.run_episode(**kwargs)

    def evaluate(self, no_sleep=False):
        self.model.noise_mode(False)
        initial, _ = self.env.reset()
        stacked_obs = deque(
            (initial for _ in range(self.n_frames)),
            maxlen=self.n_frames
        )
        rewards = 0
        while True:
            t = time.time()
            obs_tuple = tuple(stacked_obs)
            model_input = np.array([obs_tuple], dtype=np.float32)
            action = self.get_action(model_input)
            obs_next, rew, done, _, _ = self.env.step(action)
            rewards += rew
            stacked_obs.append(obs_next)
            if done:
                break
            t = self.GAP - (time.time() - t)
            if t > 0 and not no_sleep:
                time.sleep(t)
        self.model.noise_mode(True)
        print('eval reward', rewards)
        return rewards

    def learn(self):  # update with a given batch
        obs, act, rew, obs_next, done = self.replay_buffer.sample(self.batch_size)
        obs = self._preprocess(obs)
        with torch.no_grad():
            self.model.reset_noise()
            self.target_model.reset_noise()
            act = torch.as_tensor(act,
                                  dtype=torch.int64,
                                  device=self.device)
            rew = torch.as_tensor(rew,
                                  dtype=torch.float32,
                                  device=self.device)
            obs_next = self._preprocess(obs_next)
            done = torch.as_tensor(done,
                                   dtype=torch.float32,
                                   device=self.device)

        with torch.amp.autocast(self.device):
            with torch.no_grad():
                target_q = self.target_model(obs_next).detach()
                if self.is_double:
                    max_act = self.model(obs_next).detach()
                    max_act = torch.argmax(max_act, dim=1)
                    length = self.batch_size * 2 if self.transform else self.batch_size
                    max_target_q = target_q[torch.arange(length), max_act]
                    max_target_q = max_target_q.unsqueeze(-1)
                else:
                    max_target_q, _ = target_q.max(dim=1, keepdims=True)
                if self.transform:
                    max_target_q = max_target_q[:self.batch_size] + max_target_q[self.batch_size:]
                    max_target_q /= 2.
                target = rew + self.gamma * max_target_q * (1. - done)

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            q = self.model(obs)
            if self.transform:
                q = (q[:self.batch_size] + q[self.batch_size:]) / 2.
            q = torch.gather(q, 1, act)
            loss = self.criterion(q, target)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.model.eval()

        with torch.no_grad():
            loss = float(loss.detach().cpu().numpy())
            if self._learn_since_replace == self.target_steps:
                self._update_target()
                self.target_replace_times += 1
                self._learn_since_replace = 0
                if self.target_steps > 500:  # prevent frequent printing
                    print(f'target replaced {self.target_replace_times} times')
            self.learn_steps += 1
            self._learn_since_replace += 1
        return loss

    def load_explorations(self, save_loc='./explorations/'):
        assert not save_loc.endswith('\\')
        stats_imgs = []
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
            stats_imgs.append(obs_lst.flatten())
            stacked_obs = deque(
                (obs_lst[0] for _ in range(self.n_frames)),
                maxlen=self.n_frames
            )
            for o, a, r, d in zip(obs_lst[1:], action_lst, rew_lst, done_lst):
                obs_tuple = self._process_frames(stacked_obs)
                stacked_obs.append(o)
                self.replay_buffer.add(obs_tuple, a, r, d)
        stats_imgs = np.concatenate(stats_imgs)
        self.stats = [np.mean(stats_imgs), np.std(stats_imgs)]
        print('loading complete, with buffer length', len(self.replay_buffer))
        print('loaded data with mean/std', self.stats)

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

    def log(self, info, log_step):
        if not self.no_save:
            for k, v in info.items():
                self.writer.add_scalar(k, v, log_step)
