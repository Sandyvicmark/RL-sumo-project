import os
import sys
import gym
import numpy as np
from Tools import SumoSDK
from Tools.Statistics import Observer
import traci


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(CustomEnv, self).__init__()
        """state: length 9, length of waiting vehicles in every direction + phase of TLS"""
        """action: discrete 8, 8 phases of TLS"""
        self.observation_space = gym.spaces.Box(low=np.zeros((9,)),
                                                high=np.ones((9,)),
                                                shape=(9,),
                                                dtype=np.float)
        self.action_space = gym.spaces.Discrete(8)

        self.done = False

        sumo_binary = 'sumo-gui'
        sumo_cmd = [sumo_binary, '-c', '/Users/sandymark/RL-sumo/tls.sumocfg']
        traci.start(sumo_cmd)

    def reset(self, return_state=True):


    def step(self, action):
        # 1. Apply the action obtained from RL, run step()
        current_phase = traci.trafficlight.getPhase('gneJ1')
        if action != current_phase:
            traci.trafficlight.setPhase('gneJ1', action * 2 - 1)
        else:
            pass
        traci.simulationStep()

        # 2. Get new state
        s_ = self.get_state()

        # 3. Calculate reward of last action
        r = self.get_reward()

        # 4. return tuple
        return s_, r, self.done, {}

    def render(self, mode='human'):
        pass

    def get_state(self):
        pass

    def get_reward(self):
        pass