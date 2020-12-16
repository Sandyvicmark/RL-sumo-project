"""------------------------------
    Script abstract:
        This script is one which train a DQN-car to dissipate stop-and-go waves
------------------------------"""

import os
import sys
import numpy as np
import torch.nn as nn
from Tools import SumoSDK as Sumo
from Tools.Statistics import Observer
from RLlib.dqn import DQN

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

N_CAR = 40
TOTAL_STEP = 20000
STEP_LENGTH = 0.2
START_TURBULENCE = 7000
TURBULENCE_TERM = 500
TEST_TERM = 5000
ACT_SPACE = np.linspace(-3, 2, 50)
N_STATE = 4
N_ACT = ACT_SPACE.size
MEMORY_CAPACITY = 500
TARGET_UPDATE_ITER = 1000
BATCH_SIZE = 256
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
NET = nn.Sequential(nn.Linear(N_STATE, 10),
                    nn.LeakyReLU(),
                    nn.Linear(10, N_ACT))
ROUTE = ['edge2_0', 'edge3_0']
DQN_CAR = ['999']
T = 2


class Trainer:
    def __init__(self):
        self.route = []
        self.dqn_car = DQN_CAR
        self.observer = Observer()
        self.dqn = DQN(NET, N_STATE, N_ACT, MEMORY_CAPACITY, LR, EPSILON, GAMMA, TARGET_UPDATE_ITER, BATCH_SIZE)

        self.veh_list = []
        self.s = {}
        self.a = {}
        self.veh_sq = None
        self.act = {}
        self.act_old = {}
        self.reward = {}
        self.acc = {}
        self.v = {}
        self.avg_v = [0.]
        self.headway = {}
        self.emer_brake_count = {}
        self.dang_headway_count = {}
        for item in self.dqn_car:
            self.act_old[item] = 0.
            self.reward[item] = []
            self.acc[item] = []
            self.v[item] = []
            self.headway[item] = []
            self.emer_brake_count[item] = 0
            self.dang_headway_count[item] = 0
        self.step = 0
        self.av_step = 0
        self.turbulence_car = '3'
        self.speed_avail = [12, 5, 0, 14, 7, 0, 12, 4, 13, 2, 0, 14, 2, 14, 1]

    def _get_state(self, car: list):
        s = {}
        for car_ in car:
            veh_sq = Sumo.get_veh_sequence(self.route, car_, N_CAR)
            den_former, den_later = self._get_density(car_, 200, veh_sq)
            s[car_] = [traci.vehicle.getSpeed(car_),
                       veh_sq[1][0],
                       den_former,
                       den_later,
                       ]
        return s

    def _get_reward(self, car):
        if car not in traci.simulation.getCollidingVehiclesIDList():
            r = 0
            if self.s[car][1] <= 7:   # punishment for dangerous headway
                r -= 200
                if self.s[car][0] > 0:
                    r -= 500
                    r -= self.act[car] * 150
            r -= 3 * abs(self.s[car][2] - self.s[car][3]) ** 1.2  # punishment for density
        else:
            r = -10000  # punishment for collision
        return r

    def _get_pos_from_start(self):
        relative_pos = []
        for car in self.veh_list:
            pos = 0
            car_lane_index = self.route.index(traci.vehicle.getLaneID(car))
            for idx in range(car_lane_index):
                pos += traci.lane.getLength(self.route[idx])
            pos += traci.vehicle.getLanePosition(car)
            relative_pos.append(pos)
        return relative_pos

    @staticmethod
    def _get_density(car, cover_range, veh_sq):
        print('_get_density: ', car)
        n_former_car = 0
        n_later_car = 0
        former_a = 0
        former_delta_v = 0
        avg_itv_former = 0
        avg_itv_later = 0
        max_interval = 0
        for idx in range(len(veh_sq[0])):
            if veh_sq[1][idx] <= cover_range:
                n_former_car += 1
                try:
                    avg_itv_former += (veh_sq[1][idx + 1] - veh_sq[1][idx])
                    former_a += traci.vehicle.getAcceleration(veh_sq[0][idx])
                    former_delta_v += (1 + max(1 - idx / 10, 0)) * \
                                      (traci.vehicle.getSpeed(car) - traci.vehicle.getSpeed(veh_sq[0][idx]))
                except IndexError:
                    avg_itv_former = 0
            elif 628 - cover_range <= veh_sq[1][idx] < 628:
                n_later_car += 1
                try:
                    avg_itv_later += veh_sq[1][idx] - veh_sq[1][idx - 1]
                    # max_interval = max(veh_sq[1][idx + 1] - veh_sq[1][idx], max_interval)
                except IndexError:
                    avg_itv_later = 0
                    # max_interval = max(629 - veh_sq[1][idx], max_interval)
        if n_former_car:
            avg_itv_former -= (veh_sq[1][n_former_car] - veh_sq[1][n_former_car - 1])
            avg_itv_former /= max(n_former_car - 1, 1)
            if avg_itv_former:
                # den_former = min(50 * (1 / avg_itv_former ** 0.5) * n_former_car +
                #                  10 * (-former_a) + 5 * former_delta_v +
                #                  1500 / veh_sq[1][0], 500)
                den_former = min((48 - 0.24 * avg_itv_former) * n_former_car +
                                 10 * (-former_a) + 5 * former_delta_v +
                                 1500 / veh_sq[1][0], 5000)
                # 2 * (traci.vehicle.getSpeed(carID) - traci.vehicle.getSpeed(veh_sq[0][0])) + \
            else:
                den_former = min(2 * (traci.vehicle.getSpeed(car) - traci.vehicle.getSpeed(veh_sq[0][0])) +
                                 1500 / veh_sq[1][0], 5000)
        else:
            den_former = 0

        if n_later_car:
            print(avg_itv_later)
            avg_itv_later -= (veh_sq[1][len(veh_sq[0]) - n_later_car] - veh_sq[1][len(veh_sq[0]) - n_later_car - 1])
            avg_itv_later /= max(n_later_car - 1, 1)
            print(avg_itv_later)
            if avg_itv_later:
                # den_later = min(50 * (1 / avg_itv_later ** 0.5) * n_later_car +
                #                 1500 / (650 - veh_sq[1][-1]), 500)
                den_later = min((48 - 0.24 * avg_itv_later) * n_later_car +
                                1500 / (628 - veh_sq[1][-1]), 5000)
            else:
                den_later = min(1500 / (628 - veh_sq[1][-1]), 5000)
        else:
            den_later = 0
        print('avg_former: ', avg_itv_former)
        print('avg_later: ', avg_itv_later)
        print('n_former: ', n_former_car)
        print('n_later: ', n_later_car)
        print('DENSITY: ', car, den_former, den_later)
        return den_former, den_later

    def run(self):
        Sumo.wait_all_vehicles(N_CAR)
        self.s = self._get_state(self.dqn_car)
        while self.step <= TOTAL_STEP:
            for car_ in self.dqn_car:
                action = self.dqn.choose_act(self.s[car_])
                self.a[car_] = action[0]
                Sumo.apply_acceleration(car_, action, STEP_LENGTH)
            traci.simulationStep()
            for car_ in self.dqn_car:
                r = self._get_reward(car_)
                s_ = self._get_state([car_])[car_]
                self.dqn.store_transition(self.s[car_], self.a[car_], r, s_)
                self.s[car_] = s_
                self.reward[car_].append(r)
            if self.dqn.memory_counter > MEMORY_CAPACITY:
                self.dqn.learn()
            self.step += 1

            Sumo.make_turbulence('25', self.step, START_TURBULENCE, TURBULENCE_TERM, 0.15*TURBULENCE_TERM,
                                 random_scale=0.1)
            self.observer.plot_var_dyn([self.reward['999'][-1], ], self.step, 100, ['r'])
            v = []
            for car_ in self.veh_list:
                v.append(traci.vehicle.getSpeed(car_)) if traci.vehicle.getSpeed(car_) != -2 ** 30 else v.append(0.)
            pos = self._get_pos_from_start()
            self.observer.plot_stop_and_go_wave(N_CAR, pos, v, self.step, 500, TOTAL_STEP, 1000)


