import gc
import gym
import cv2
import time
import enum
import random
import pyautogui
import threading
import numpy as np
from mss.windows import MSS as mss

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.


class Actions(enum.Enum):
    @classmethod
    def random(cls):
        return random.choice(list(cls))


class Move(Actions):
    # NO_OP = 0
    HOLD_LEFT = 1
    HOLD_RIGHT = 2
    LOOK_LEFT = 3
    LOOK_RIGHT = 4


class Attack(Actions):
    NO_OP = 0
    ATTACK = 1
    UP_ATTACK = 2
    SPELL = 3


class JumpDash(Actions):
    NO_OP = 0
    TIMED_SHORT_JUMP = 1
    TIMED_LONG_JUMP = 2
    DASH = 3


class HKEnv(gym.Env):
    KEYMAPS = {
        Move.HOLD_LEFT: 'a',
        Move.HOLD_RIGHT: 'd',
        Move.LOOK_LEFT: 'a',
        Move.LOOK_RIGHT: 'd',
        JumpDash.TIMED_SHORT_JUMP: 'space',
        JumpDash.TIMED_LONG_JUMP: 'space',
        JumpDash.DASH: 'k',
        Attack.ATTACK: 'j',
        Attack.UP_ATTACK: ('w', 'j'),
        Attack.SPELL: 'q'
    }
    HP_CKPT = [64, 99, 135, 171, 207, 242, 278, 314, 352]
    ACTIONS = [Move, Attack, JumpDash]

    def __init__(self, obs_shape=(160, 160), w1=1., w2=1., w3=0.002):
        self.monitor = self._find_window()
        self.holding = []
        self.prev_knight_hp = None
        self.prev_enemy_hp = None
        self.prev_action = -1
        total_actions = np.prod([len(Act) for Act in self.ACTIONS])
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                dtype=np.uint8, shape=obs_shape)
        self.action_space = gym.spaces.Discrete(int(total_actions))

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self._timer = None
        self._episode_time = None

    @staticmethod
    def _find_window():
        window = pyautogui.getWindowsWithTitle('Hollow Knight')
        assert len(window) == 1, f'found {len(window)} windows called Hollow Knight {window}'
        window = window[0]
        try:
            window.activate()
        except Exception:
            window.minimize()
            window.maximize()
            window.restore()
        window.resizeTo(1280, 720)
        window.moveTo(0, 0)
        geo = None
        while geo is None:
            geo = pyautogui.locateOnScreen('./locator/geo.png', confidence=0.9)
            time.sleep(0.2)
        loc = {
            'left': geo.left - 48,
            'top': geo.top - 78,
            'width': 986,
            'height': 640
        }
        return loc

    def _timed_hold(self, key, seconds):
        def timer_thread():
            pyautogui.keyDown(key)
            time.sleep(seconds)
            pyautogui.keyUp(key)
            time.sleep(0.001)

        if self._timer is None or not self._timer.is_alive():
            # timer available, do timed action
            # ignore if there is already a timed action in progress
            self._timer = threading.Thread(target=timer_thread)
            self._timer.start()
            return 0
        else:
            return 1

    def _step_actions(self, actions):
        for key in self.holding:
            pyautogui.keyUp(key)
        self.holding = []
        action_rew = 0
        for act in actions:
            if not act.value:
                continue
            key = self.KEYMAPS[act]

            if act.name.startswith('HOLD'):
                pyautogui.keyDown(key)
                self.holding.append(key)
            elif act.name.startswith('TIMED'):
                action_rew += self._timed_hold(key, act.value * 0.2)
            elif isinstance(key, tuple):
                with pyautogui.hold(key[0]):
                    pyautogui.press(key[1])
            else:
                pyautogui.press(key)
        return action_rew * -1e-5

    def _to_multi_discrete(self, num):
        num = int(num)
        chosen = []
        for Act in self.ACTIONS:
            num, mod = divmod(num, len(Act))
            chosen.append(Act(mod))
        return chosen

    def _find_menu(self):
        monitor = self.monitor
        monitor = (monitor['left'] + monitor['width'] // 2,
                   monitor['top'],
                   monitor['width'] // 2,
                   monitor['height'] // 2)
        return pyautogui.locateOnScreen(f'locator/menu_badge.png',
                                        region=monitor,
                                        confidence=0.85)

    def observe(self, knight_only=False):
        with mss() as sct:
            frame = np.asarray(sct.grab(self.monitor), dtype=np.uint8)[:, :, :3]
        # print(frame.shape)
        if knight_only:
            enemy_hp = None
        else:
            enemy_hp_bar = frame[625, 191:769, :3]
            enemy_hp_bar_red = enemy_hp_bar[..., 2]
            enemy_hp = (((enemy_hp_bar_red - enemy_hp_bar[..., 0]) == 201)
                        & ((enemy_hp_bar_red - enemy_hp_bar[..., 1]) == 209))
            enemy_hp = enemy_hp.sum()
            if enemy_hp == 0:
                enemy_hp = len(enemy_hp_bar)
            enemy_hp /= len(enemy_hp_bar)
        knight_hp_bar = frame[45, :353, 0]
        knight_hp = (knight_hp_bar[self.HP_CKPT] > 180).sum()
        if knight_only:
            obs = None
        else:
            obs = frame[:608, ...]
            obs = cv2.resize(obs,
                             dsize=self.observation_space.shape,
                             interpolation=cv2.INTER_AREA)
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        return obs, knight_hp, enemy_hp

    def step(self, actions):
        action_rew = 0
        if actions == self.prev_action:
            action_rew -= 2e-5
        self.prev_action = actions
        actions = self._to_multi_discrete(actions)
        action_rew += self._step_actions(actions)
        obs, knight_hp, enemy_hp = self.observe()

        win = self.prev_enemy_hp < enemy_hp
        lose = knight_hp == 0
        done = win or lose

        if win:
            lose = False
            enemy_hp = 0.
        hurt = knight_hp < self.prev_knight_hp
        hit = enemy_hp < self.prev_enemy_hp

        reward = (
                - self.w1 * hurt
                + self.w2 * hit
                + action_rew
        )
        if not (hurt or hit):
            reward += self.w3
        if win:  # extra reward for winning based on conditions
            time_rew = 5. / (time.time() - self._episode_time)
            reward += knight_hp / 40. + time_rew
        elif lose:
            reward -= enemy_hp / 5.
        # print('reward', reward)
        # print()

        self.prev_knight_hp = knight_hp
        self.prev_enemy_hp = enemy_hp
        reward = np.clip(reward, -1.5, 1.5)
        return obs, reward, done, False, None

    def reset(self, seed=None, options=None):
        super(HKEnv, self).reset(seed=seed)
        self.cleanup()
        while True:
            if self._find_menu():
                break
            pyautogui.press('w')
            time.sleep(0.75)
        pyautogui.press('space')

        # wait for loading screen
        ready = False
        while True:
            obs, _, _ = self.observe()
            is_loading = (obs < 20).sum() < 10
            if ready and not is_loading:
                break
            else:
                ready = is_loading
        time.sleep(2.25)
        self.prev_knight_hp, self.prev_enemy_hp = len(self.HP_CKPT), 1.
        self._episode_time = time.time()
        return self.observe()[0], None

    def close(self):
        self.cleanup()

    def cleanup(self):
        if self._timer is not None:
            self._timer.join()
        self.holding = []
        for key in self.KEYMAPS.values():
            pyautogui.keyUp(key)
        self.prev_knight_hp = None
        self.prev_enemy_hp = None
        self.prev_action = -1
        self._timer = None
        self._episode_time = None
        gc.collect()
