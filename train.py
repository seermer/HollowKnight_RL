import torch
import numpy as np
from torch.backends import cudnn

import models
import time
import hkenv

cudnn.benchmark = True


@torch.no_grad()
def main():
    game = hkenv.HKEnv((160, 160))
    m = models.SimpleExtractor((160, 160), 5)
    m = models.SinglePathMLP(m, game.total_actions).cuda()
    m.eval()
    # warmup
    m(torch.rand((2, 5, 160, 160), dtype=torch.float32, device='cuda'))
    m(torch.rand((1, 5, 160, 160), dtype=torch.float32, device='cuda'))
    buffer = []
    game.start()
    while True:
        # print('iter')
        t = time.time()
        if len(buffer) >= 5:
            frames = np.array([buffer[-5:]], dtype=np.float16)
            frames = torch.cuda.FloatTensor(frames)
            pred = m(frames).detach().cpu().view(-1)
            obs, _, done = game.step_pred(pred)
            if done:
                game.cleanup()
                break
        else:
            obs, _, _ = game.observe()
        buffer.append(obs)
        print(time.time() - t)


if __name__ == '__main__':
    main()
