#!/usr/bin env python3

import numpy as np
import stable_baselines3 as sb3
from for_stable_baselines.CustomEnvTest import CustomEnv
import matplotlib.pyplot as plt
import torch
# import tensorflow as tf

plt.ion()

env = CustomEnv()
policy_kwargs = dict(act_fun=torch.nn.functional.relu, net_arch=[20, 20])
model = sb3.PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.001, ent_coef=0, n_steps=512)
# model = sb.DQN('MlpPolicy', env, gamma=0.9, learning_rate=0.01, buffer_size=500, batch_size=256, double_q=False,
#                target_network_update_freq=1000)
model.learn(total_timesteps=30000)
plt.figure(1)
plt.plot(np.linspace(0, env.n_step, env.n_step), env.record_a)
plt.show()
plt.figure(2)
plt.plot(np.linspace(0, env.n_step, env.n_step), env.record_r)
plt.show()


plt.ioff()
