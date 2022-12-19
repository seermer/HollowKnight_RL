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
    NO_OP = 0
    HOLD_LEFT = 1
    HOLD_RIGHT = 2


class Jump(Actions):
    NO_OP = 0
    TIMED_SHORT_JUMP = 1
    TIMED_LONG_JUMP = 2


class Attack(Actions):
    NO_OP = 0
    ATTACK = 1


class Dash(Actions):
    NO_OP = 0
    DASH = 1


class HKEnv(gym.Env):
    KEYMAPS = {
        Move.HOLD_LEFT: 'a',
        Move.HOLD_RIGHT: 'd',
        Jump.TIMED_SHORT_JUMP: 'space',
        Jump.TIMED_LONG_JUMP: 'space',
        Dash.DASH: 'k',
        Attack.ATTACK: 'j',
    }
    HP_CKPT = [64, 99, 135, 171, 207, 242, 278, 314, 352]
    ACTIONS = [Move, Jump, Attack]

    def __init__(self, obs_shape=(160, 160), w1=1., w2=1., w3=0.):
        self.monitor = self._find_window()
        self.holding = []
        self.prev_knight_hp = None
        self.prev_enemy_hp = None
        total_actions = np.prod([len(Act) for Act in self.ACTIONS])
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                dtype=np.uint8, shape=obs_shape)
        self.action_space = gym.spaces.Discrete(int(total_actions))

        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self._w3 = w3

        self._timer = None

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
        loc = {
            'left': window.left + 160,
            'top': window.top + 64,
            'width': 986,
            'height': window.height - 80
        }
        return loc

    def _timed_hold(self, key, seconds):
        def timer_thread():
            pyautogui.keyDown(key)
            time.sleep(seconds)
            pyautogui.keyUp(key)
            time.sleep(0.0025)

        if self._timer is None or not self._timer.is_alive():
            # timer available, do timed action
            # ignore if there is already a timed action in progress
            self._timer = threading.Thread(target=timer_thread)
            self._timer.start()

    def _step_actions(self, actions):
        for key in self.holding:
            pyautogui.keyUp(key)
        self.holding = []

        for act in actions:
            if not act.value:
                continue
            key = self.KEYMAPS[act]
            if act.name.startswith('HOLD'):
                pyautogui.keyDown(key)
                self.holding.append(key)
            elif act.name.startswith('TIMED'):
                self._timed_hold(key, act.value * 0.2)
            else:
                pyautogui.press(key)

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
                                        confidence=0.9)

    def observe(self):
        with mss() as sct:
            frame = np.asarray(sct.grab(self.monitor), dtype=np.uint8)[:, :, :3]
        # print(frame.shape)
        enemy_hp_bar = frame[625, 191:769, :3]
        enemy_hp_bar_red = enemy_hp_bar[..., 2]
        # print(enemy_hp_bar[..., 2] - enemy_hp_bar[..., 1], enemy_hp_bar[..., 2] - enemy_hp_bar[..., 0])
        enemy_hp = (((enemy_hp_bar_red - enemy_hp_bar[..., 0]) == 201)
                    & ((enemy_hp_bar_red - enemy_hp_bar[..., 1]) == 209))
        enemy_hp = enemy_hp.sum()
        if enemy_hp == 0:
            enemy_hp = len(enemy_hp_bar)
        enemy_hp /= len(enemy_hp_bar)
        # print(enemy_hp)
        knight_hp_bar = frame[45, :353, 0]
        knight_hp = (knight_hp_bar[self.HP_CKPT] > 180).sum()
        obs = frame[:608, ...]
        obs = cv2.resize(obs,
                         dsize=self.observation_space.shape,
                         interpolation=cv2.INTER_AREA)
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        return obs, knight_hp, enemy_hp

    def step(self, actions):
        actions = self._to_multi_discrete(actions)
        self._step_actions(actions)
        time.sleep(0.025)
        obs, knight_hp, enemy_hp = self.observe()
        if self.prev_knight_hp is None:
            self.prev_knight_hp = knight_hp
            self.prev_enemy_hp = enemy_hp
            return obs, self.w3, False, False, {}

        win = self.prev_enemy_hp < enemy_hp
        lose = knight_hp == 0
        done = win or lose
        if enemy_hp < self.prev_enemy_hp:  # enemy gets hit
            self.w3 = self._w3
        else:  # nothing happens
            self.w3 -= 0.00012
            self.w3 = max(self.w3, -self._w3)
        if knight_hp < self.prev_knight_hp:  # knight gets hit
            self.w3 = min(self.w3, 0.)
        if win:
            enemy_hp = 0
        reward = (
                self.w1 * np.sign(knight_hp - self.prev_knight_hp)
                + self.w2 * np.sign(self.prev_enemy_hp - enemy_hp)
                + self.w3
        )
        if win:  # extra reward for winning based on remaining health
            reward += np.log10(knight_hp / 2.)
        # print('reward', reward)
        # print()
        # TODO: add special reward for winning with shorter steps/time

        self.prev_knight_hp = knight_hp
        self.prev_enemy_hp = enemy_hp
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super(HKEnv, self).reset(seed=seed)
        self.cleanup()
        while True:
            if self._find_menu():
                break
            pyautogui.press('w')
            time.sleep(0.1)
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
        time.sleep(0.5)
        return self.observe()[0], {}

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
        self.w3 = self._w3
        self._timer = None
