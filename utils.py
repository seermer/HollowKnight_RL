from collections import deque
from functools import partial
import keyboard
import numpy as np
import time

release_jump = False

def move(left=True):
    if left:
        keyboard.send("right", do_press=False)
        keyboard.send("left", do_release=False)
    else:
        keyboard.send("left", do_press=False)
        keyboard.send("right", do_release=False)


def release_moves():
    keyboard.send("right", do_press=False)
    keyboard.send("left", do_press=False)

def xjump():
    global release_jump
    partial(keyboard.send, hotkey="space", do_release=False)
    release_jump = False


ACTIONS = (("left", "right", "--"), ("xjump", "sjump", "-"), ("attack", "uattack", "dash", "-"))
FRAMES = deque(maxlen=5)
KEYMAP = {"sjump": partial(keyboard.send, hotkey="space"),
          "xjump": xjump,
          "left": partial(move, left=True),
          "right": partial(move, left=False),
          "attack": partial(keyboard.send, hotkey="j"),
          "uattack": partial(keyboard.send, hotkey="up,j"),
          "dash": partial(keyboard.send, hotkey="shift"),
          "--": release_moves,
          "-": partial(time.sleep, secs=.02)}
MASK = np.zeros((40, 100, 3), np.uint8)
