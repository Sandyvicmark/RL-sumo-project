import math
import os
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

EPISODE = 100000
TURBULENCE_SEQUENCE = [7, 4, 8, 5, 7, 4, 9, 5, 8, 5, 9, 4, 8, 3, 0, 5, 0, 7, 0, 8, 1, 9, 0]
START_TURBULENCE = 50000
TURBULENCE_TERM = 500

T = 1.5
ROUTE1 = ['edge2_0', 'edge3_0']


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-3, high=2, shape=(1,), dtype=np.float)
        # self.action_space = gym.spaces.Discrete(50)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0, -10]),
                                                high=np.array([15, 200 * np.pi, 10, 15]),
                                                dtype=np.float)
        self.s = np.array([])
        self.n_step = 0
        self.done_step = 0
        self.n_turbulence = 0
        self.done = False
        self.record_a = []
        self.record_r = []

        sumoBinary = 'sumo-gui'
        sumoCmd = [sumoBinary, '-c', '/home/sandymark/RL-sumo/net4stable_baselines3.sumocfg']
        traci.start(sumoCmd)

        veh_sq = self._get_veh_sequence('999', ROUTE1)
        interval = self._get_interval('999', ROUTE1, veh_sq[0])
        """
        s: current state
        index:  0: speed of dqn-car
                1: headway
                2: speed of leader
                3: difference of dqn-car and leader
        """
        self.s = np.array([traci.vehicle.getSpeed('999'), interval, traci.vehicle.getSpeed(veh_sq[0]),
                           traci.vehicle.getSpeed('999') - traci.vehicle.getSpeed(veh_sq[0])])
        self.act_old = 0.

    def reset(self):
        self.done = False
        veh_sq = self._get_veh_sequence('999', ROUTE1)
        interval = self._get_interval('999', ROUTE1, veh_sq[0])
        s = np.array([traci.vehicle.getSpeed('999'), interval, traci.vehicle.getSpeed(veh_sq[0]),
                      traci.vehicle.getSpeed('999') - traci.vehicle.getSpeed(veh_sq[0])])
        return s

    def step(self, action):
        while True:
            veh_list = traci.vehicle.getIDList()
            # print('VEH_LIST: ', veh_list)
            if len(veh_list) != 10:
                traci.simulationStep()
            else:
                break

        # a = np.clip(action, -3, 2)
        # a = (action - 30) / 10.
        # print(type(a))
        # print('a: ', a[0])
        traci.vehicle.setSpeed('999', max(0, self.s[0] + 0.1 * action[0]))
        traci.simulationStep()

        veh_sq = self._get_veh_sequence('999', ROUTE1)
        r = self._get_reward('999', action[0], veh_sq)
        interval = self._get_interval('999', ROUTE1, veh_sq[0])
        self.s = np.array([traci.vehicle.getSpeed('999'), interval, traci.vehicle.getSpeed(veh_sq[0]),
                           traci.vehicle.getSpeed('999') - traci.vehicle.getSpeed(veh_sq[0])])
        self.act_old = action[0]

        self.record_a.append(action[0])
        self.record_r.append(r)
        self.n_step += 1

        if self.n_step >= START_TURBULENCE and self.n_step % TURBULENCE_TERM == 0:
            traci.vehicle.setSpeed('0', TURBULENCE_SEQUENCE[self.n_turbulence % len(TURBULENCE_SEQUENCE)])
            self.n_turbulence += 1
        # if self.n_step % 500 == 0:
        #     plt.plot(np.linspace(0, self.n_step, self.n_step), self.record_a)

        if '999' in traci.simulation.getCollidingVehiclesIDList() or self.done_step >= 1000:
            self.done = True
            self.done_step = 0
        else:
            self.done_step += 1

        if self.n_step > 0 and self.n_step % 500 == 0:
            plt.cla()
            plt.figure(1)
            plt.plot(np.linspace(0, self.n_step, self.n_step), self.record_r, 'b')
            plt.xlim(self.n_step - 10000 if self.n_step > 10000 else 0, self.n_step)
            plt.ylim(-6000, 500)
            plt.figure(2)
            plt.plot(np.linspace(0, self.n_step, self.n_step), self.record_a, 'r')
            plt.xlim(self.n_step - 10000 if self.n_step > 10000 else 0, self.n_step)
            plt.pause(0.1)
        return self.s, r, False, {}

    def render(self, mode='human'):
        pass

    @staticmethod
    def _get_veh_sequence(carID, route):
        """ Generate a list, storing the car sequence before dqn-car,
            in which the closest former car is the first element. """
        while True:
            try:
                veh_list = []
                for lane in route:
                    veh_list += traci.lane.getLastStepVehicleIDs(lane)
                veh_sequence = veh_list[veh_list.index(carID) + 1:] + veh_list[: veh_list.index(carID)]
                break
            except ValueError:
                traci.simulationStep()
                continue
        return veh_sequence

    @staticmethod
    def _get_interval(carID, route, leader_carID):
        car_lane_index = route.index(traci.vehicle.getLaneID(carID))
        leader_car_lane_index = route.index(traci.vehicle.getLaneID(leader_carID))
        if leader_car_lane_index < car_lane_index:
            leader_car_lane_index += len(route)
        itv = 0.
        for n_lane in range(car_lane_index, leader_car_lane_index):
            itv += traci.lane.getLength(route[n_lane % len(route)])
        itv -= traci.vehicle.getLanePosition(carID)
        itv += traci.vehicle.getLanePosition(leader_carID)
        if abs(itv) <= 1e-6:
            itv = 1e-6
        if itv < 0:
            for lane in route:
                itv += traci.lane.getLength(lane)
        return itv

    def _get_reward(self, carID, act, veh_sequence):
        if carID not in traci.simulation.getCollidingVehiclesIDList():  # 3. Calculate the reward
            s_star = 5 + 3 + max(0, self.s[0] * T + self.s[0] * (self.s[3]) / (2 * math.sqrt(2. * 3.)))
            a_star = min(2, max(-3, 2. * (1 - (self.s[0] / 15.) ** 4 - (s_star / self.s[1]) ** 2)))
            # print('a_star: ', carID, a_star)
            r = -500 * abs(act - a_star)  # for a

            # if ((traci.vehicle.getAcceleration(veh_sq[0]) <= -2 and s[car][1] <= 20) or
            #     (s[car][3] >= 5 and s[car][1] <= (s[car][0]**2 - s[car][2]**2) / 2*2.5 + 3)) and act[car] > -2:
            #     r -= 1000
            #
            # if s[car][1] <= 7 and act[car] >= -2:
            #     r -= 1000
            # elif s[car][1] <= 7 and act[car] < -2:
            #     r += 500

            if (traci.vehicle.getAcceleration(veh_sequence[0]) <= -2 and self.s[1] <= 20) or \
                    (self.s[3] >= 5 and self.s[1] <= (self.s[0] ** 2 - self.s[2] ** 2) / 2 * 2.5 + 3):  # or \
                # s[1] <= 7:
                if act > -2:
                    # print('NO EMER BREAKING!!')
                    r -= 3000
                else:
                    r += 1000                  # for essential emergency break

            if self.s[1] <= 7:
                r -= 500
                # print('DANGEROUS ZOOM!!')
                if self.s[0] > 0:
                    # print('ACTING------')
                    # r -= (100 + act * 40)
                    r -= 2000
                    if act > 0:
                        r -= 3000
                    elif act <= 0:
                        r += abs(act) * 1000
            # print('r: ', r)

                # r -= 2. / (s[1] - 4.8) * 1000  # for dangerous headway (new collision punishment)

            # r -= min(400, abs(s[3]) ** 4)
            # r -= min(50, 10 * abs(act - self.act_old))  # for delta a
        else:
            r = -10000
        return r



# env = gym.make('CartPole-v1')
# model = sb.PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=10000)

