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
    def __init__(self, env, replay_buffer,
                 n_frames, gamma, eps, eps_func, target_steps, learn_freq,
                 model, lr, lr_decay, criterion, batch_size, device,
                 is_double=True, drq=True, svea=True, reset=0, n_targets=1,
                 save_suffix='', no_save=False):
        """
        Initialize a DQN trainer

        :param env: any gym environment (pixel based recommended but not required)
        :param replay_buffer: a Buffer instance (or its subclass)
        :param n_frames: number of consecutive frames to input into the model
        :param gamma: discount factor
        :param eps: epsilon for epsilon-greedy, 0 means no epsilon-greedy
        :param eps_func: a function being called with eps, num_step,
                where eps is previous epsilon, num_step is number of environment steps,
                should return new epsilon (this function is for epsilon decay)
        :param target_steps: number of steps that target model being replaced,
                note that this step is based on number of learning steps not env steps
        :param learn_freq: learning frequency (for example,
                4 means update online network every 4 environment steps)
        :param model: an instance of AbstractFullyConnected subclass
        :param lr: learning rate
        :param lr_decay: True if lr decays over time after first target replce
        :param criterion: loss function that used should take online Q and target Q as input
        :param batch_size: number of samples to learn for each update
        :param device: device to put the model, only CUDA GPU is supported!!!!
        :param is_double: True to use double DQN update
        :param drq: True to use Data regularized Q
        :param svea: True to use svea, only work with drq enabled
        :param reset: number of environment steps between each model reset (0 for no reset)
        :param n_targets: number of targets to use for Averaged-DQN, 1 for no averaged
        :param save_suffix: suffix for save location
        :param no_save: True if avoid saving logs and models
        """
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

        assert n_targets > 0, 'only positive number of targets supported'
        self.model = model.to(device)
        self.target_models = [copy.deepcopy(self.model) for _ in range(n_targets)]
        self.init_lr = lr
        self.final_lr = lr * 0.625
        self.lr_decay = lr_decay
        self.optimizer = torch.optim.NAdam(self.model.parameters(),
                                           lr=lr, eps=0.005 / batch_size)
        self.model.eval()
        for target in self.target_models:
            target.eval()
            for param in target.parameters():
                param.requires_grad = False
        self.criterion = criterion
        if hasattr(self.criterion, 'to'):
            self.criterion = self.criterion.to(device)
        if hasattr(self.criterion, 'reduction'):
            self.criterion.reduction = 'none'
        self.batch_size = batch_size
        assert device == 'cuda', 'this version of code only supports cuda, ' \
                                 'so that mixed precision GradScaler can be used'
        self.device = device
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()

        self.is_double = is_double
        if drq:
            assert len(self.env.observation_space.shape) == 3
        pad = tuple(np.array(self.env.observation_space.shape[1:], dtype=int) // 20)
        self.transform = K.RandomCrop(size=self.env.observation_space.shape[1:],
                                      padding=pad,
                                      padding_mode='replicate').to(device) if drq else None
        self.svea = svea
        self.reset = reset
        if svea:
            assert drq, 'svea can only work with drq enabled'

        self.steps = 0
        self.learn_steps = 0
        self.target_replace_times = 0
        self._learn_since_replace = 0

        self.no_save = no_save
        save_loc = ('./saved/'
                    + str(int(time.time()))
                    + save_suffix)
        assert not save_loc.endswith('\\')
        self.save_loc = save_loc if save_loc.endswith('/') else f'{save_loc}/'
        if not no_save:
            print('save files at', self.save_loc)
            if not os.path.exists(self.save_loc):
                os.makedirs(self.save_loc)
            self.writer = SummaryWriter(self.save_loc + 'log/')

        for _ in range(3):
            self._warmup(self.model)
            for target in self.target_models:
                self._warmup(target)

    @staticmethod
    def _process_frames(frames):
        return tuple(frames)

    @staticmethod
    def _rescale(obs):
        obs /= 127.5
        obs -= 1.
        return obs

    @staticmethod
    def _save_transitions(obs_lst, action_lst, rew_lst, done_lst, fname):
        assert len(obs_lst) - 1 == len(action_lst) == len(rew_lst) == len(done_lst)
        assert isinstance(obs_lst[0], np.ndarray)
        assert not os.path.exists(fname)
        obs_lst = np.array(obs_lst, dtype=obs_lst[0].dtype)
        if max(action_lst) < 256:
            action_lst = np.array(action_lst, dtype=np.uint8)
        else:
            action_lst = np.array(action_lst, dtype=np.uint32)
        rew_lst = np.array(rew_lst, dtype=np.float32)
        done_lst = np.array(done_lst, dtype=np.bool8)
        np.savez_compressed(fname, o=obs_lst, a=action_lst, r=rew_lst, d=done_lst)

    @torch.no_grad()
    def _preprocess_train_obs(self, obs, no_transform=False, cat_orig=False):
        obs = torch.as_tensor(obs,
                              device=self.device)
        if len(obs.shape) != 4:  # not image
            return obs

        obs = self._rescale(obs)
        if self.transform and not no_transform:
            scale = torch.randn((self.batch_size, 1, 1, 1),
                                device=self.device)
            scale = torch.clip_(scale, -2, 2) * 0.05 + 1.
            augmented = torch.vstack((obs * scale, self.transform(obs)))
            augmented = torch.clip_(augmented, -1, 1)
            if cat_orig:
                augmented = torch.vstack((augmented, obs))
            return augmented
        else:
            return obs

    def _update_target(self, i: int):
        if i < len(self.target_models):
            self.target_models[i].load_state_dict(self.model.state_dict())
            self.target_models[i].eval()
            if i == 0:
                self.target_replace_times += 1
                self._learn_since_replace = 0
                if self.target_steps > 500:
                    print(f'target replaced {self.target_replace_times} times')

    def _warmup(self, model):
        """
        pytorch is very slow on first run,
        warmup to reduce impact on actual training
        """
        c, *shape = self.env.observation_space.shape
        with torch.amp.autocast(self.device):
            model(torch.rand((self.batch_size, self.n_frames * c) + tuple(shape),
                             device=self.device)).detach().cpu().numpy()

    @torch.no_grad()
    def _compute_target(self, obs_next, rew, done):
        with torch.amp.autocast(self.device):
            obs_next = self._preprocess_train_obs(obs_next,
                                                  no_transform=self.svea,
                                                  cat_orig=False)
            rew = torch.as_tensor(rew,
                                  device=self.device)
            done = torch.as_tensor(done,
                                   device=self.device)
            target_q = self.target_models[0](obs_next)
            for target in self.target_models[1:]:
                target_q += target(obs_next)
            if len(self.target_models) > 1:
                target_q /= len(self.target_models)
            if self.is_double:
                with torch.inference_mode():
                    max_act = self.model(obs_next, adv_only=True)
                    max_act = torch.argmax(max_act, dim=-1, keepdim=True)
                max_target_q = torch.gather(target_q, -1, max_act)
            else:
                max_target_q, _ = target_q.max(dim=-1, keepdims=True)
            if self.transform and not self.svea:
                max_target_q = max_target_q[:self.batch_size] + max_target_q[self.batch_size:]
                max_target_q /= 2.
            target = rew + self.gamma * max_target_q * (1. - done)
        return target.detach()

    @torch.inference_mode()
    def get_action(self, obs):
        """
        find the action with largest Q value output by online model
        """
        with torch.amp.autocast(self.device):
            obs = torch.as_tensor(obs,
                                  device=self.device).unsqueeze(0)
            if len(obs.shape) == 4:
                self._rescale(obs)
            pred = self.model(obs, adv_only=True).cpu().numpy()[0]
        return np.argmax(pred)

    def run_episode(self, random_action=False, cache=False):
        """
        run an episode with policy, and learn between steps

        :param random_action: whether to always do random actions
                (for exploration, will not update network while this is True)
        :param cache: whether to save the transitions locally to disk
        :return: total episode reward, per sample loss, and current learning rate
        """
        save_loc = self.save_loc + 'transitions/'
        if cache:
            if not os.path.exists(save_loc):
                os.makedirs(save_loc)
            fname = f'{int(time.time())}.npz'
            i = 0
            while os.path.exists(save_loc + fname):
                fname = f'{int(time.time())}_{i}.npz'
                i += 1
            save_loc += fname
            print('caching episode transitions to\n', os.path.abspath(save_loc))

        if self.lr_decay and not random_action and self.target_replace_times:
            # decay lr over first 300 episodes
            decay = (self.init_lr - self.final_lr) / 300.
            for group in self.optimizer.param_groups:
                group['lr'] = max(self.final_lr,
                                  group['lr'] - decay)
        initial, _ = self.env.reset()
        stacked_obs = deque(
            (initial for _ in range(self.n_frames)),
            maxlen=self.n_frames
        )

        obs_lst = [initial]
        action_lst, rew_lst, done_lst = [], [], []

        total_rewards = 0
        total_loss = 0
        learned_times = 0
        obs_next_tuple = self._process_frames(stacked_obs)
        while True:
            obs_tuple = obs_next_tuple
            if random_action or self.eps > random.uniform(0, 1):
                action = self.env.action_space.sample()
            else:
                model_input = np.concatenate(obs_tuple, dtype=np.float32)
                action = self.get_action(model_input)
            obs_next, rew, done, _, _ = self.env.step(action)
            total_rewards += rew
            self.steps += 1
            if cache:
                obs_lst.append(obs_next)
                action_lst.append(action)
                rew_lst.append(rew)
                done_lst.append(done)
            stacked_obs.append(obs_next)
            obs_next_tuple = self._process_frames(stacked_obs)
            self.replay_buffer.add(obs_tuple, action, rew, done, obs_next_tuple)
            if self.reset and self.steps % self.reset == 0:
                print('model reset')
                self.model.reset_params()
                for i in range(len(self.target_models)):
                    self._update_target(i)
            if not random_action:
                self.eps = self.eps_func(self.eps, self.steps)
                if len(self.replay_buffer) > self.batch_size and self.steps % self.learn_freq == 0:
                    # print(self.num_batches)
                    for _ in range(self.num_batches):
                        total_loss += self.learn()
                        learned_times += 1
            if done:
                break
        if not random_action:
            self.replay_buffer.step()
        if cache:
            self._save_transitions(obs_lst, action_lst, rew_lst, done_lst, save_loc)
        avg_loss = total_loss / learned_times if learned_times > 0 else 0
        return total_rewards, avg_loss, self.optimizer.param_groups[0]['lr']

    def run_episodes(self, n, **kwargs):
        for _ in range(n):
            self.run_episode(**kwargs)

    def evaluate(self):
        """
        evaluate the current policy greedily
        """
        self.model.noise_mode(False)
        initial, _ = self.env.reset()
        stacked_obs = deque(
            (initial for _ in range(self.n_frames)),
            maxlen=self.n_frames
        )
        rewards = 0
        while True:
            obs_tuple = tuple(stacked_obs)
            model_input = np.concatenate(obs_tuple, dtype=np.float32)
            action = self.get_action(model_input)
            obs_next, rew, done, _, _ = self.env.step(action)
            rewards += rew
            stacked_obs.append(obs_next)
            if done:
                break
        self.model.noise_mode(True)
        print('eval reward', rewards)
        return rewards

    def learn(self):  # update with a given batch
        """
        sample a single batch and update current model

        :return: per sample loss
        """
        if self.replay_buffer.prioritized:
            (obs, act, rew, obs_next, done), indices = \
                self.replay_buffer.prioritized_sample(self.batch_size)
        else:
            obs, act, rew, obs_next, done = \
                self.replay_buffer.sample(self.batch_size)
            indices = None
        self.model.reset_noise()
        for target in self.target_models:
            target.reset_noise()

        target = self._compute_target(obs_next, rew, done)
        act = torch.as_tensor(act,
                              dtype=torch.int64,
                              device=self.device)
        self.model.train()
        with torch.amp.autocast(self.device):
            obs = self._preprocess_train_obs(obs,
                                             no_transform=False,
                                             cat_orig=self.svea)
            obs.requires_grad = True
            self.optimizer.zero_grad(set_to_none=True)
            q = self.model(obs)
            if self.transform:
                if self.svea:
                    q = (q[:self.batch_size] +
                         q[self.batch_size:self.batch_size * 2] +
                         q[self.batch_size * 2:])
                    q /= 3.
                else:
                    q = (q[:self.batch_size] + q[self.batch_size:])
                    q /= 2.
            q = torch.gather(q, -1, act)
            loss = self.criterion(q, target)
            if self.replay_buffer.prioritized:
                with torch.no_grad():
                    error = q.detach() - target
                    error = error.cpu().numpy().flatten()
                    error = np.abs(error)
                    weights = self.replay_buffer.update_priority(error, indices)
                weights = torch.as_tensor(weights, device=self.device)
                loss *= weights.reshape(loss.shape)
            loss = loss.mean()

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.model.eval()

        with torch.no_grad():
            loss = float(loss.detach().cpu().numpy())
            self._learn_since_replace += 1
            self.learn_steps += 1
            self._update_target(self._learn_since_replace % self.target_steps)
        return loss

    def load_explorations(self, save_loc='./explorations/'):
        """
        load all explorations from given environment into the replay buffer

        :param save_loc: directory where the explorations can be found
        """
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
            obs_next_tuple = self._process_frames(stacked_obs)
            for o, a, r, d in zip(obs_lst[1:], action_lst, rew_lst, done_lst):
                obs_tuple = obs_next_tuple
                stacked_obs.append(o)
                obs_next_tuple = self._process_frames(stacked_obs)
                self.replay_buffer.add(obs_tuple, a, r, d, obs_next_tuple)
        print('loading complete, with buffer length', len(self.replay_buffer))

    def save_explorations(self, n_episodes, save_loc='./explorations/'):
        """
        interact with environment n episodes with random agent, save locally
        please note that explorations are only effective
        when the environment is exactly the same
        (other settings, including replay buffer can change)

        this function will automatically skip any existing explorations,
        manually delete any explorations you do not want to keep

        :param n_episodes: number of explorations to be saved
        :param save_loc: directory used to save
        """
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
                action = self.env.action_space.sample()
                obs_next, rew, done, _, _ = self.env.step(action)
                obs_lst.append(obs_next)
                action_lst.append(action)
                rew_lst.append(rew)
                done_lst.append(done)
                if done:
                    break
            self._save_transitions(obs_lst, action_lst, rew_lst, done_lst, fname)
            print(f'saved exploration at {os.path.abspath(fname)}')
            print(f'total reward {np.sum(rew_lst)}')

    def save_models(self, prefix='', online_only=False):
        """
        save online model, target models, and optimizer checkpoint

        :param prefix: prefix for the model file you want to save
        :param online_only: True if only save online
        """
        if not self.no_save:
            torch.save(self.model.state_dict(), self.save_loc + prefix + 'online.pt')
            if online_only:
                return
            for i, target in enumerate(self.target_models):
                torch.save(target.state_dict(), self.save_loc + prefix + f'target{i}.pt')
            torch.save(self.optimizer.state_dict(), self.save_loc + prefix + 'optimizer.pt')

    def log(self, info, log_step):
        """
        log given dictionary info to tensorboard

        :param info: a dictionary of tag (key) and scaler (value) pairs
        :param log_step: the step to be logged on tensorboard
        """
        if not self.no_save:
            for k, v in info.items():
                self.writer.add_scalar(k, v, log_step)
