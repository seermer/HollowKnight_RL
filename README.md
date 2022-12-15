# Playing Hollow Knight with Reinforcement Learning

The project uses Deep Q-Network to learn how to play boss fight in Hollow Knight.

I am currently primarily trying to train the agent with Hornet boss fight, but theoretically, it should be able to run on most Hall of Gods Boss with only one health bar.

You need to install the Satchel and EnemyHPBar Mod to correctly run (so it can recognize boss HP), I have made some modifications (custom HP bar color) for more stable recognition, I will be updating the files on github soon.

Project Tentative Plan: <br>
- [ ] Update Project Structure
  - [ ] Add requirements.txt, mod files
  - [ ] Add instructions to train
  - [ ] Add more docstrings and typehint in code
- [ ] Train an agent that can defeat Hornet
- [ ] add code for additional functionalities (like saving buffer, model...)
- [ ] improve the algorithm


Project inspired by https://github.com/ailec0623/DQN_HollowKnight <br>
that is a very interesting project, and the author has already defeated Hornet with a trained agent. However, that project uses CE and windows API heavily, which I am less familiar with, so I decided to make one on my own.
