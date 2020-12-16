import os
import sys
import numpy as np
import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Tools.Statistics import Observer, Logger


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

STEP_LENGTH = 0.2
ROUTE_LENGTH = 671.
N_CAR = 40
EPISODE = 20000
START_TURBULENCE = 300
TURBULENCE_TERM = 300
TEST_TERM = 2500
EMISSION_TEST_TERM = EPISODE
ACT_SPACE = np.linspace(-3, 2, 51)
N_STATE = 5
N_ACT = ACT_SPACE.size
MEMORY_CAPACITY = 500
TARGET_UPDATE_ITER = 1000
BATCH_SIZE = 256
LR = [0.01, 0.005, 0.003]
EPSILON = 0.95
GAMMA = 0.95
REWARD_TOLERANCE = [-400, -300, -200]
REWARD_OK_RATIO = [0.85, 0.9, 0.9]
LEARN_SW = True

T = 1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(N_STATE, 10)           # build hidden layer
        self.hidden1.weight.data.normal_(0, 0.1)        # initialize params of hidden layer
        # self.hidden2 = nn.Linear(20, 20)  # build hidden layer
        # self.hidden2.weight.data.normal_(0, 0.1)  # initialize params of hidden layer
        self.out = nn.Linear(10, N_ACT)                 # build output layer
        self.out.weight.data.normal_(0, 0.1)            # initialize params of output layer

    def forward(self, x):                               # forward
        x = self.hidden1(x)
        x = F.leaky_relu(x)
        # x = self.hidden2(x)
        # x = F.leaky_relu(x)
        act_value = self.out(x)
        return act_value


class DQN:
    def __init__(self):
        use_old_nn = int(input('Use trained network? '))
        if use_old_nn:
            d = int(input('Input network number: '))
            self.eval_net = tc.load('/Users/sandymark/RL-sumo/net/evalnet' + str(d) + '.torch')
            self.target_net = tc.load('/Users/sandymark/RL-sumo/net/targetnet' + str(d) + '.torch')
        else:
            self.eval_net, self.target_net = Net(), Net()
        # self.eval_net.cuda()
        # self.target_net.cuda()

        self.update_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATE*2 + 2))
        self.optimizer = tc.optim.Adam(self.eval_net.parameters(), lr=LR[0])
        self.loss_func = nn.MSELoss()
        self.record_loss = [None]

    def choose_act(self, x):
        x = tc.unsqueeze(tc.FloatTensor(x), 0)  # .cuda()
        if np.random.uniform() <= EPSILON:
            act_value = self.eval_net.forward(x)  # .cpu()
            action = tc.max(act_value, 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, N_ACT)
        return action

    def store_transition(self, s, a, r, s_):
        trans = np.hstack((s, [a, r], s_))
        if trans.shape != (N_STATE*2 + 2,):
            print('WARNING::trans shape ERROR!!!')
        else:
            index = self.memory_counter % MEMORY_CAPACITY
            self.memory[index, :] = trans
            self.memory_counter += 1

    def learn(self):
        if self.update_step_counter % TARGET_UPDATE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.update_step_counter += 1

        for i in range(5):
            sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
            batch_memory = self.memory[sample_index, :]
            b_s = tc.FloatTensor(batch_memory[:, :N_STATE])  # .cuda()
            b_a = tc.LongTensor(batch_memory[:, N_STATE:N_STATE + 1].astype(int))  # .cuda()
            b_r = tc.FloatTensor(batch_memory[:, N_STATE + 1:N_STATE + 2])  # .cuda()
            b_s_ = tc.FloatTensor(batch_memory[:, -N_STATE:])  # .cuda()

            """ 
                Input current STATE to 'eval_net', the net outputs the q_value of each action.
                Use .gather(1, b_a) to select the q_value of the action that we really chose 
                by e-greedy above, where the 1st arg of gather means search by col(0) or row(1), 
                2nd arg is an index, to search the value we want to pick up. 
            """
            q_eval = self.eval_net(b_s).gather(1, b_a)

            """ 
                .detach() means resist back-propagation from the 'target_net', 
                because we only need to update the 'target_net' in some special steps.
            """
            q_next = self.target_net(b_s_).detach()

            """ 
                .max() 'll return a tuple, where 1st element is value and 2nd element is its index. 
            """
            q_target = b_r + GAMMA * tc.unsqueeze(q_next.max(1)[0], 0).t()

            if q_eval.size() != q_target.size():
                print('shape of q_eval: ', q_eval.shape, 'shape of q_target: ', q_target.shape,
                      'shape of q_next: ', q_next.shape)
                print('shape of b_r: ', b_r.shape)
                print('q_next.max(1)[0]: ', q_next.max(1)[0], q_next.max(1)[0].shape)
                raise AssertionError
            loss = self.loss_func(q_eval, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.record_loss.append(float(loss))


class Train:
    def __init__(self):
        self.route = route1
        self.dqn_car = dqn_car
        self.observer = Observer()
        self.logger = Logger()
        self.reset_pos_x = [69.13818384165964, 66.57395614066074, 62.37045669318575, 56.63118960624632, 49.49747468305833, 41.14496766047312, 31.779334981768276, 21.631189606246323, 10.950412552816164, 0.0, -10.950412552816173, -21.631189606246313, -31.77933498176827, -41.144967660473114, -49.49747468305832, -56.631189606246316, -62.370456693185744, -66.57395614066074, -69.13818384165964, -70.0, -69.13818384165964, -66.57395614066074, -62.370456693185766, -56.63118960624633, -49.497474683058336, -41.14496766047313, -31.779334981768283, -21.63118960624633, -10.950412552816173, 0.0, 10.950412552816147, 21.631189606246306, 31.779334981768265, 41.14496766047311, 49.497474683058314, 56.631189606246316, 62.370456693185744, 66.57395614066074, 69.13818384165964, 70.0]
        self.reset_pos_y = [210.95041255281615, 221.63118960624632, 231.77933498176827, 241.14496766047313, 249.49747468305833, 256.6311896062463, 262.37045669318576, 266.57395614066075, 269.1381838416596, 270.0, 269.1381838416596, 266.57395614066075, 262.37045669318576, 256.6311896062463, 249.49747468305833, 241.14496766047313, 231.77933498176827, 221.63118960624632, 210.95041255281618, 200.0, 189.04958744718385, 178.36881039375365, 168.22066501823176, 158.85503233952687, 150.50252531694167, 143.36881039375368, 137.62954330681424, 133.42604385933925, 130.86181615834036, 130.0, 130.86181615834036, 133.42604385933925, 137.62954330681424, 143.36881039375368, 150.50252531694167, 158.85503233952687, 168.22066501823173, 178.36881039375368, 189.04958744718382, 200.0]
        self.reset_cross_posx = [250.0, 251.38150398011618, 255.4496737905816, 261.97970171999845, 270.61073738537635, 280.86582838174553, 292.17827674798843, 303.92295478639227, 315.45084971874735, 326.1249282357974, 335.3553390593274, 342.63200821770465, 347.5528258147577, 349.8458666866564, 349.3844170297569, 346.19397662556435, 340.45084971874735, 332.4724024165092, 322.69952498697734, 311.67226819279523,  250.0, 248.61849601988382, 244.5503262094184, 238.02029828000155, 229.38926261462365, 219.1341716182545, 207.82172325201154, 196.07704521360776, 184.54915028125262, 173.87507176420257, 164.64466094067262, 157.36799178229538, 152.44717418524232, 150.1541333133436, 150.6155829702431, 153.80602337443565, 159.54915028125262, 167.5275975834908, 177.30047501302266, 188.32773180720474]

        self.reset_cross_posy = [150.0, 138.3277318072047, 127.30047501302266, 117.5275975834908, 109.54915028125262, 103.80602337443565, 100.61558297024311, 100.1541333133436, 102.44717418524232, 107.36799178229539, 114.64466094067262, 123.87507176420257, 134.54915028125262, 146.07704521360776, 157.82172325201154, 169.1341716182545, 179.38926261462365, 188.02029828000153, 194.5503262094184, 198.61849601988382,  250.0, 261.6722681927953, 272.69952498697734, 282.47240241650917, 290.45084971874735, 296.19397662556435, 299.3844170297569, 299.8458666866564, 297.5528258147577, 292.6320082177046, 285.3553390593274, 276.1249282357974, 265.45084971874735, 253.92295478639227, 242.17827674798846, 230.86582838174553, 220.61073738537635, 211.97970171999847, 205.4496737905816, 201.38150398011618]


        self.veh_list = []
        self.s = {}
        self.a = {}
        self.veh_sq = None
        self.act = {}
        self.act_old = {}
        self.reward = {}
        self.acc = {}
        self.v = {}
        self.avg_v = {}
        self.headway = {}
        self.fuel = []
        self.emer_brake_count = {}
        self.dang_headway_count = {}
        self.ok_flag = False
        for item in dqn_car:
            self.act_old[item] = 0.
            self.reward[item] = []
            self.acc[item] = []
            self.v[item] = []
            self.headway[item] = []
            self.emer_brake_count[item] = 0
            self.dang_headway_count[item] = 0
        self.step = 0
        self.av_step = 0
        self.train_stage = 0
        self.last_stage_step = 0
        self.epsilon_select = [0.97, 0.99, 1]
        self.turbulence_car = []
        self.statistic_car = []
        self.speed_avail = [12, 5, 0, 14, 7, 0, 12, 4, 13, 2, 0, 14, 2, 14, 1]

        self.csv_dir = '/Users/sandymark/RL-sumo/Output data/test/'

    def _get_veh_sequence(self):
        """ Generate a list, storing the car sequence before dqn-car,
            in which the closest former car is the first element. """
        while True:
            try:
                while True:
                    veh_list = [[], []]
                    for lane in self.route:
                        veh_list[0] += traci.lane.getLastStepVehicleIDs(lane)
                    if len(veh_list[0]) != N_CAR:
                        traci.simulationStep()
                        continue
                    else:
                        abs_pos = self._get_absolute_pos(veh_list[0])
                        veh_list[1] = abs_pos
                        # for item in veh_list[0]:
                        #     veh_list[1].append(self._get_interval(carID, item))
                        # print('veh_list: ', veh_list)
                        break
                break
            except ValueError:
                traci.simulationStep()
                print('ValueError')
                continue

        return veh_list

    # def _get_interval(self, carID, leader_carID):
    #     car_lane_index = self.route.index(traci.vehicle.getLaneID(carID))
    #     leader_car_lane_index = self.route.index(traci.vehicle.getLaneID(leader_carID))
    #     if leader_car_lane_index < car_lane_index:
    #         leader_car_lane_index += len(self.route)
    #     itv = 0.
    #     for n_lane in range(car_lane_index, leader_car_lane_index):
    #         itv += traci.lane.getLength(self.route[n_lane % len(self.route)])
    #     itv -= traci.vehicle.getLanePosition(carID)
    #     itv += traci.vehicle.getLanePosition(leader_carID)
    #     if abs(itv) <= 1e-6:
    #         itv = 1e-6
    #     if itv < 0:
    #         for lane in self.route:
    #             itv += traci.lane.getLength(lane)
    #     return itv

    @staticmethod
    def _get_interval_test(pos_cur_car, veh_sq):
        veh_sq[1] -= pos_cur_car * np.ones((len(veh_sq[1])))
        for idx in range(len(veh_sq[1])):
            if veh_sq[1][idx] < 0:
                veh_sq[1][idx] += ROUTE_LENGTH  # 629
        return list(veh_sq[1])

    def _get_state(self, carID, veh_sq_):
        den_former, den_later, vhf, vhl, nf, nl = self._get_density(veh_sq_)
        s = [traci.vehicle.getSpeed(carID),
             veh_sq_[1][0],
             # 629 - veh_sq_[1][-1],
             ROUTE_LENGTH - veh_sq_[1][-1],
             traci.vehicle.getSpeed(carID) - traci.vehicle.getSpeed(veh_sq_[0][0]),
             den_former - den_later,
             # vhf,
             # vhl,
             # nf,
             # nl,
             ]
        return s
        # s = []
        # for idx in range(len(self.veh_list)):
        #     s.append(traci.vehicle.getSpeed(self.veh_list[idx]))
        # pos = self._get_absolute_pos()
        # pack = zip(s, pos)
        # pack = sorted(pack, key=lambda x: x[1])
        # print('pack: ', pack)
        # s_, pos_ = zip(*pack)
        # return s_ + pos_

    @staticmethod
    def _get_density(veh_sq):
        look_range = 200
        former_car_list = []
        later_car_list = []
        for idx in range(len(veh_sq[0])):
            if 0 <= veh_sq[1][idx] < look_range:
                former_car_list.append(veh_sq[0][idx])
            elif ROUTE_LENGTH - look_range < veh_sq[1][idx] < ROUTE_LENGTH:
                later_car_list.append(veh_sq[0][idx])

        n_former_car = len(former_car_list)
        n_later_car = len(later_car_list)

        v_h_for = 1e-6
        v_h_lat = 1e-6
        sum_d_for = 0
        sum_d_lat = 0
        w_v_f = 0
        w_v_l = 0
        w_p_f = 0
        w_p_l = 0
        for idx, car_ in zip(range(len(former_car_list)), former_car_list):
            # v_h_for += 1 / max(0.1, traci.vehicle.getSpeed(car_))  # Harmonic Sum
            # v_h_for += max(0.1, traci.vehicle.getSpeed(car_))  # Arithmetic Sum
            v_h_for += (1 - math.pow(veh_sq[1][idx] / 400, 1)) / max(0.1, traci.vehicle.getSpeed(car_))  # Weighted Harmonic Sum
            w_v_f += (1 - math.pow(veh_sq[1][idx] / 400, 1))
            # sum_d_for += (1 - traci.vehicle.getAcceleration(car_) / 3) / veh_sq[1][idx]
            # w_p_f += (1 - traci.vehicle.getAcceleration(car_) / 3)
            if idx == 0:
                sum_d_for += 1 / veh_sq[1][0]
                # w_p_f += 200
                w_p_f += 1
            else:
                sum_d_for += (1 - math.pow(veh_sq[1][idx - 1] / 400, 1)) / max(0.1, veh_sq[1][idx] - veh_sq[1][idx - 1])
                # w_p_f += (200 - veh_sq[1][idx - 1])
                w_p_f += (1 - math.pow(veh_sq[1][idx - 1] / 400, 1))
        # v_h_for = max(1 / v_h_for, 0.1)  # Harmonic Mean
        # v_h_for = max(v_h_for / n_former_car, 0.1)  # Arithmetic Mean
        v_h_for = max(w_v_f / v_h_for, 0.1)  # Weighted Harmonic Mean
        for idx, car_ in zip(range(len(later_car_list)), later_car_list):
            # v_h_lat += 1 / max(0.1, traci.vehicle.getSpeed(car_))
            # v_h_lat += max(0.1, traci.vehicle.getSpeed(car_))
            v_h_lat += (1 - math.pow((ROUTE_LENGTH - veh_sq[1][idx - n_later_car]) / 400, 1)) / max(0.1, traci.vehicle.getSpeed(car_))
            w_v_l += (1 - math.pow((ROUTE_LENGTH - veh_sq[1][idx - n_later_car]) / 400, 1))
            # sum_d_lat += (1 - traci.vehicle.getAcceleration(car_) / 3) / (veh_sq[1][idx - n_later_car + 1] - veh_sq[1][idx - n_later_car])
            # w_p_l += (1 - traci.vehicle.getAcceleration(car_) / 3)
            if idx == n_later_car - 1:
                sum_d_lat += 1 / (ROUTE_LENGTH - veh_sq[1][-1])
                # w_p_l += 200
                w_p_l += 1
            else:
                sum_d_lat += (1 - math.pow((ROUTE_LENGTH - veh_sq[1][idx - n_later_car + 1]) / 400, 1)) / max(0.1, veh_sq[1][idx - n_later_car + 1] - veh_sq[1][idx - n_later_car])
                # w_p_l += (200 - (629 - veh_sq[1][idx - n_later_car + 1]))
                w_p_l += (1 - math.pow((ROUTE_LENGTH - veh_sq[1][idx - n_later_car + 1]) / 400, 1))
        # v_h_lat = max(1 / v_h_lat, 0.1)
        # v_h_lat = max(v_h_lat / n_later_car, 0.1)
        v_h_lat = max(w_v_l / v_h_lat, 0.1)
        if n_former_car:
            # den_former = n_former_car + 5 * (1 - math.pow(veh_sq[1][0] / 600, 1)) / v_h_for + 1000 / max(0.1, veh_sq[1][0])
            den_former = 300 * sum_d_for / w_p_f + 15 / v_h_for + 500 / max(0.1, veh_sq[1][0])
        else:
            den_former = 0
        if n_later_car:
            # den_later = n_later_car + 5 * (1 - math.pow((629 - veh_sq[1][-1]) / 600, 1)) / v_h_lat + 1000 / max(0.1, 629 - veh_sq[1][-1])
            den_later = 300 * sum_d_lat / w_p_l + 15 / v_h_lat + 500 / max(0.1, (ROUTE_LENGTH - veh_sq[1][-1]))
        else:
            den_later = 0

        print('den_former: ', den_former, 'sdf: ', sum_d_for, 'wpf: ', w_p_f, 'vh: ', v_h_for, 'itv: ', veh_sq[1][0])
        print('den_later: ', den_later, 'sdll: ', sum_d_lat, 'wpl: ', w_p_l, 'vh: ', v_h_lat, 'itv: ', ROUTE_LENGTH - veh_sq[1][-1])
        return min(2000., den_former*2.5), min(2000., den_later*2.5), v_h_for, v_h_lat, n_former_car, n_later_car
        # return n_former_car*50, n_later_car * 50

    def _get_reward(self, carID):
        r = 0                                                       # 3. Calculate the reward
        # s_star = 5 + 3 + max(0, self.s[carID][0] * T + self.s[carID][0] * (self.s[carID][2]) / (2 * math.sqrt(2. * 3.)))
        # a_star = min(2, max(-3, 2. * (1 - (self.s[carID][0] / 15.) ** 4 - (s_star / self.s[carID][1]) ** 2)))
        # print('a_star: ', carID, a_star)
        # r = -150 * abs(self.act[carID] - a_star)  # for a

        # if ((traci.vehicle.getAcceleration(veh_sq[0]) <= -2 and s[car][1] <= 20) or
        #     (s[car][3] >= 5 and s[car][1] <= (s[car][0]**2 - s[car][2]**2) / 2*2.5 + 3)) and act[car] > -2:
        #     r -= 1000
        #
        # if s[car][1] <= 7 and act[car] >= -2:
        #     r -= 1000
        # elif s[car][1] <= 7 and act[car] < -2:
        #     r += 500

        # if (traci.vehicle.getAcceleration(veh_sequence[0][0]) <= -2 and self.s[carID][1] <= 20) or \
        #         (self.s[carID][3] >= 5 and self.s[carID][1] <= (self.s[carID][0] ** 2 - traci.vehicle.getSpeed(veh_sequence[0][0]) ** 2) / 2 * 2.5 + 3):  # or \
        #     # s[1] <= 7:
        #     if self.act[carID] > -2:
        #         r -= 2000
        #     # else:
        #     #     r += 500                    # for essential emergency break

        if self.s[carID][1] <= 7:
            r -= 500
            if self.s[carID][0] > 0:
                r -= 500
                r -= self.act[carID] * 150
                # if act > 0:
                #     r -= 1000
                # elif act <= 0:
                #     r += abs(act) * 400
            # r -= 15 / max(0.1, self.s[carID][0]) * 100  # for dangerous headway (new collision punishment)

        # r -= min(400, abs(s[3]) ** 4)
        r -= 100 * abs(self.act[carID] - self.act_old[carID])  # for delta a
        # r += self.s[carID][-1] * 500    # for avg_v
        # r -= abs(self.act[carID]) * 300     # for fuel
        r -= 30 * abs(self.s[carID][4])  # for density
        r -= 80 * abs(self.act[carID]) ** 2
        # if carID in traci.simulation.getCollidingVehiclesIDList():
        # if self.s[carID][1] <= 5:
        #     r -= 10000
        return max(-20000, r)

    def _get_absolute_pos(self, veh_sq):
        abs_pos = []
        for car_ in veh_sq:
            car_lane_index = self.route.index(traci.vehicle.getLaneID(car_))
            pos = 0
            for idx in range(car_lane_index):
                pos += traci.lane.getLength(self.route[idx])
            pos += traci.vehicle.getLanePosition(car_)
            # if car_lane_index == 0:
            #     pos = traci.vehicle.getLanePosition(car_)
            # elif car_lane_index == 1:
            #     pos = traci.lane.getLength(self.route[0]) + traci.vehicle.getLanePosition(car_)
            abs_pos.append(pos)
        return abs_pos

    # def _get_reward_test(self, carID):
    #     v = []
    #     v_des = 9.5
    #     hdway_des = 650 / N_CAR
    #     sigma_hdway = 0
    #     veh_sq = self._get_veh_sequence(carID)
    #     for car in self.veh_list:
    #         v.append(traci.vehicle.getSpeed(car))
    #     v = np.array(v)
    #     sigma_hdway += max(hdway_des - veh_sq[1][0], 0)
    #     for idx in range(1, len(veh_sq[0])):
    #         sigma_hdway += max(hdway_des - (veh_sq[1][idx] - veh_sq[1][idx - 1]), 0)
    #     sigma_hdway += max(hdway_des - (650 - veh_sq[1][-1]), 0)
    #     # if carID in traci.simulation.getCollidingVehiclesIDList():
    #     #     collision = -1000
    #     # else:
    #     #     collision = 0
    #     return np.linalg.norm(v_des * np.ones((N_CAR,))) - np.linalg.norm(v_des * np.ones((N_CAR,)) - v) #- sigma_hdway

    def dynamic_lr_change(self, policy):
        global EPSILON
        if self.step > self.last_stage_step + 3000:
            n_ok_r = 0
            for r_ in self.reward['999'][-500:]:
                if r_ >= REWARD_TOLERANCE[self.train_stage]:
                    n_ok_r += 1
            if n_ok_r / 500 > REWARD_OK_RATIO[self.train_stage] or \
               self.step >= self.last_stage_step + 10000:
                EPSILON = self.epsilon_select[self.train_stage]
                policy.optimizer = tc.optim.Adam(policy.eval_net.parameters(), lr=LR[self.train_stage])
                self.train_stage += 1
                self.last_stage_step = self.step
            if self.train_stage > 2:
                self.ok_flag = True
        print('#############################\n')
        print('TRAIN STAGE: ', self.train_stage)

    def reset(self):
        veh_list = sorted(self.veh_list, key=lambda x: int(x))
        for idx in range(len(veh_list)):
            if idx < 20:
                traci.vehicle.moveToXY(veh_list[idx], 'edge2', 0, self.reset_pos_x[idx], self.reset_pos_y[idx])
            else:
                traci.vehicle.moveToXY(veh_list[idx], 'edge3', 0, self.reset_pos_x[idx], self.reset_pos_y[idx])
            traci.vehicle.setSpeed(veh_list[idx], 0)
        traci.simulationStep()
        for car_ in veh_list:
            traci.vehicle.setSpeed(car_, -1)

    def reset_cross(self):
        veh_list = sorted(self.veh_list, key=lambda x: int(x))
        traci.trafficlight.setPhase('gneJ8', 4)
        traci.simulationStep()
        for idx in range(len(veh_list)):
            if idx < 20:
                traci.vehicle.moveToXY(veh_list[idx], '82', 0, self.reset_cross_posx[idx], self.reset_cross_posy[idx])
            else:
                traci.vehicle.moveToXY(veh_list[idx], '85', 0, self.reset_cross_posx[idx], self.reset_cross_posy[idx])
            traci.vehicle.setSpeed(veh_list[idx], 0)
        traci.simulationStep()
        for car_ in veh_list:
            traci.vehicle.setSpeed(car_, -1)

    def run(self):
        global EPSILON
        dqn = DQN()
        self.veh_list = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','999']
        # traci.simulationStep()
        #
        # while True:                                                         # Waiting for all cars drive into route
        #     traci.simulationStep()
        #     self.veh_list = list(traci.vehicle.getIDList())
        #     print('VEH_LIST: ', self.veh_list)
        #     if len(self.veh_list) != N_CAR:
        #         for car in self.veh_list:
        #             traci.vehicle.setSpeed(car, 1.5)
        #         continue
        #     else:
        #         for car in self.veh_list:
        #             traci.vehicle.setSpeed(car, -1)
        #         break
        if route1 == ['edge2_0', 'edge3_0']:
            self.reset()
        else:
            self.reset_cross()
        self.turbulence_car = self.veh_list[:]
        self.statistic_car = self.veh_list[:]
        for _ in self.dqn_car:
            self.turbulence_car.remove(_)
            self.statistic_car.remove(_)
        self.statistic_car.remove('25')
        for _ in self.statistic_car:
            self.avg_v[_] = []

        self.fuel = [0.]                  # Initialize fuel gauge

        veh_sq = self._get_veh_sequence()                                        # Generate initial state
        for car_ in dqn_car:
            car_idx = veh_sq[0].index(car_)
            veh_sq_ = [veh_sq[0][car_idx + 1:] + veh_sq[0][: car_idx],
                       veh_sq[1][car_idx + 1:] + veh_sq[1][: car_idx]]
            veh_sq_[1] = self._get_interval_test(veh_sq[1][car_idx], veh_sq_)
            self.s[car_] = self._get_state(car_, veh_sq_)

        while self.step < EPISODE:                                               # Train start:
            for car_ in dqn_car:
                self.a[car_] = dqn.choose_act(self.s[car_])                                 # 1. Choose an action
                self.act[car_] = ACT_SPACE[self.a[car_]]
                # print('a: ', car_, self.act)
                # print 'get lane ID: ', car_, traci.vehicle.getLaneID(car_)
                # print('v: ', car_, self.s[car_][0])
                # print 'Lane Position: ', car_, traci.vehicle.getLanePosition(car_)
                traci.vehicle.setSpeed(car_, max(0, self.s[car_][0] + STEP_LENGTH * self.act[car_]))     # 2. Perform this action

            traci.simulationStep()
            if traci.simulation.getCollidingVehiclesNumber():
                if route1 == ['edge2_0', 'edge3_0']:
                    self.reset()
                else:
                    self.reset_cross()

            # v_sum = 0
            # for car_ in self.veh_list:
            #     if car_ not in dqn_car:
            #         v_sum += traci.vehicle.getSpeed(car_) if traci.vehicle.getSpeed(car_) >= 0 else 0
            # self.avg_v.append(v_sum / len(self.veh_list))
            for car_ in self.statistic_car:
                self.avg_v[car_].append(traci.vehicle.getSpeed(car_))

            veh_sq = self._get_veh_sequence()
            for car_ in dqn_car:
                car_idx = veh_sq[0].index(car_)
                veh_sq_ = [veh_sq[0][car_idx + 1:] + veh_sq[0][: car_idx],
                           veh_sq[1][car_idx + 1:] + veh_sq[1][: car_idx]]
                veh_sq_[1] = self._get_interval_test(veh_sq[1][car_idx], veh_sq_)
                # print('a_target: ', car_, traci.vehicle.getAcceleration(car_))
                next_tls = traci.vehicle.getNextTLS(car_)
                # print('NEXT_TLS: ', car_, next_tls)
                r = 0
                """
                    Check if the car is under the control of SUMO (due to TLS...)
                    If so, just update state AND do NOT save data into experience pool, to avoid RL learning wrong data.
                """
                if (round(traci.vehicle.getAcceleration(car_), 3) != round(self.act[car_] * STEP_LENGTH, 3)) \
                        and next_tls and next_tls[0][3] == 'r':
                    print(car_, 'a_target: ', traci.vehicle.getAcceleration(car_))
                    print(car_, 'a_act: ', self.act[car_] * STEP_LENGTH)
                    self.reward[car_].append(0)
                    self.acc[car_].append(traci.vehicle.getAcceleration(car_))
                    self.v[car_].append(self.s[car_][0])
                    self.headway[car_].append(self.s[car_][1])

                    s_ = self._get_state(car_, veh_sq_)
                    self.s[car_] = s_
                    self.act_old[car_] = traci.vehicle.getAcceleration(car_)
                else:
                    r = self._get_reward(car_)
                    if LEARN_SW:
                        self.dynamic_lr_change(dqn)
                    self.reward[car_].append(r)
                    self.acc[car_].append(self.act[car_])
                    self.v[car_].append(self.s[car_][0])
                    self.headway[car_].append(self.s[car_][1])
                    # print('REWARD: ', car_, r)

                    s_ = self._get_state(car_, veh_sq_)                            # 4. Get the next state from SUMO

                    dqn.store_transition(self.s[car_], self.a[car_], r, s_)

                    self.act_old[car_] = self.act[car_]
                    self.s[car_] = s_

                    if self.step >= EPISODE - TEST_TERM:                                  # Bad behaviour count
                        EPSILON = 1.
                        if self.act[car_] < -2:
                            self.emer_brake_count[car_] += 1
                        if self.s[car_][1] < 7:
                            self.dang_headway_count[car_] += 1

                    print('---------------------------------\n---REWARD: ',
                          r,
                          '\n---------------------------------')

            if LEARN_SW and dqn.memory_counter > MEMORY_CAPACITY:                             # 6. Train the eval-net
                dqn.learn()

            self.step += 1
            self.av_step += 1

            v = []
            cur_fuel_sum = 0
            for car_ in self.veh_list:
                if self.step < EMISSION_TEST_TERM:
                    cur_fuel_sum += traci.vehicle.getFuelConsumption(car_) * STEP_LENGTH
            #     v.append(traci.vehicle.getSpeed(car_)) if traci.vehicle.getSpeed(car_) != -2 ** 30 else v.append(0.)
            # pos = self._get_absolute_pos(self.veh_list)
            # self.observer.plot_stop_and_go_wave(N_CAR, pos, v, self.step, 0, 10000, 700)
            cur_fuel_sum /= 1000
            cur_fuel_sum += self.fuel[-1]
            self.fuel.append(cur_fuel_sum)

            """ Manually change the leader's behaviour to train the dqn-car """
            if self.step % TURBULENCE_TERM == 0 and self.step >= START_TURBULENCE:
            # if self.step == 300:
                # if self.turbulence_car:
                #     traci.vehicle.setSpeed(self.turbulence_car, -1)
                # self.turbulence_car = self.veh_list[25] \
                #     if self.turbulence_car != self.veh_list[25] else self.veh_list[20 + np.random.randint(-3, 3)]
                turbulence_car = self.turbulence_car[np.random.choice(len(self.turbulence_car))]
                traci.vehicle.setSpeed('25', 0.5)  # 0.01 + np.random.random() / 2)

            elif self.step % TURBULENCE_TERM == 20 and self.step >= START_TURBULENCE:
            # elif self.step == 320 and self.step >= START_TURBULENCE:
                traci.vehicle.setSpeed('25', -1)

            # self.observer.plot_var_dyn(var=[self.reward['999'][-1], self.avg_v['0'][-1],
            #                                 # 0 if dqn.record_loss[-1] is None else min(10000, dqn.record_loss[-1]),
            #                                 traci.vehicle.getAcceleration('5'), self.fuel[-1]],
            #                            step=self.step,
            #                            plot_term=100,
            #                            plot_range=[0, 0, 0, 0],
            #                            plot_num=0,
            #                            color=['b', 'g', 'r', 'black'])

            print('---------------------------------\n---VALUE_LOSS: ',
                  dqn.record_loss[-1],
                  '\n---------------------------------')
            if self.ok_flag:
                break
            if not LEARN_SW:
                odom = traci.vehicle.getDistance('0')
                print('ODOM: ', odom)
                if odom >= 2000 + 440:
                    break

        # print('-' * 20, '\naverage_v STD: ', np.array(self.avg_v[int(-0.3*len(self.avg_v)):]).std(), '\n', '-' * 20)
        # print('average_v MEAN: ', np.array(self.avg_v[int(-0.3*len(self.avg_v)):]).mean(), '\n', '-' * 20)
        fuel_csv_name = str(input('Input csv name: '))
        self.logger.save_csv(self.fuel, 'fuel', self.csv_dir + 'fuel.csv')
        self.logger.save_csv(self.avg_v, 'avg_v', self.csv_dir + 'avg_v.csv')
        return self.reward, self.acc, self.v, self.headway, self.fuel, self.emer_brake_count, self.dang_headway_count, \
            self.av_step, dqn.eval_net, dqn.target_net


if __name__ == '__main__':
    sumoBinary = 'sumo-gui'
    sumoCmd = [sumoBinary, '-c', '/Users/sandymark/RL-sumo/net.sumocfg', '--collision.action', 'warn']
    traci.start(sumoCmd)

    plt.ion()
    dqn_car = ['9','19','29','999']  #  ,  '998', '997', '996'], '4', '14', '24', '34'
    route1 = ['80_0', '81_0', '82_0', '83_0', '84_0', '85_0']
    # route1 = ['edge2_0', 'edge3_0']
    trainer = Train()
    re, a, vel, hdway, fuel, e_b, d_h, t, eval_net, target_net = trainer.run()
    plt.ioff()
    plt.show()
    fuel_sum = fuel[-1]
    print('Fuel consumption in total(L): ', fuel_sum)

    for i in range(len(dqn_car)):
        plt.figure(i)
        plt.plot(np.linspace(0, t, t), re[dqn_car[i]], 'b')
        plt.plot(np.linspace(0, t, t), a[dqn_car[i]], 'r')
        plt.plot(np.linspace(0, t, t), vel[dqn_car[i]], 'g')
        plt.plot(np.linspace(0, t, t), hdway[dqn_car[i]], 'y')
        plt.show()
        a_part = np.array(a[dqn_car[i]][500:])
        a_std = a_part.std()
        print('a_STD: ', dqn_car[i], a_std)
        print('Emergency brake performed: ', e_b[dqn_car[i]])
        print('Dangerous headway: ', d_h[dqn_car[i]])

    save = int(input('Save this NN? (1/0)\n'))
    if save:
        num = int(input('Input the number: '))
        tc.save(eval_net, '/Users/sandymark/RL-sumo/net/evalnet' + str(num) + '.torch')
        tc.save(target_net, '/Users/sandymark/RL-sumo/net/targetnet' + str(num) + '.torch')

    traci.close()
