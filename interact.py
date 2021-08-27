from PIL import Image
import cv2
import mss
import numpy as np
import win32gui
import keyboard
from functools import partial
from tensorflow.keras import optimizers

import agent
import model
import replay_buffer
import utils




def get_screen(sct):
    sct_img = sct.grab(win32gui.GetWindowRect(win32gui.FindWindow(None, "Hollow Knight")))
    img = cv2.resize(np.array(Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX"))[60:-30, 135:-135],
                     (260, 260),
                     interpolation=cv2.INTER_LINEAR)
    img[:40, :100] = utils.MASK
    return img


def get_action_key(nums):
    return utils.ACTIONS[0][nums[0]], utils.ACTIONS[1][nums[1]], utils.ACTIONS[2][nums[2]]


def main():
    sct = mss.mss()
    dqn = agent.DQN(get_model=partial(model.get_model, in_shape=(260, 260, 3), frames=4, out_shape=10, reduced=True),
                    buffer=replay_buffer.ReplayBuffer(10000),
                    discount=.99,
                    optimizer=optimizers.Adam(1.5e-4),
                    target_replace_step=10000)
    done = False
    while True:
        img = get_screen(sct)
        if utils.release_jump:
            keyboard.release(utils.KEYMAP["jump"])
            utils.release_jump = False
        if len(utils.FRAMES) != utils.FRAMES.maxlen:
            utils.FRAMES.append(img)
            continue
        for action in dqn.get_action(np.array(utils.FRAMES[1:])):
            utils.KEYMAP[action]()


if __name__ == '__main__':
    main()
