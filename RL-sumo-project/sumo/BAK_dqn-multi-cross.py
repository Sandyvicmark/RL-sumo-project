import os
import sys
import numpy as np
import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Tools.Statistics import Observer

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

N_CAR = 40
EPISODE = 20000
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

T = 2


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
            self.eval_net = tc.load('/home/sandymark/RL-sumo/net/evalnet' + str(d) + '.torch')
            self.target_net = tc.load('/home/sandymark/RL-sumo/net/targetnet' + str(d) + '.torch')
        else:
            self.eval_net, self.target_net = Net(), Net()
        # self.eval_net.cuda()
        # self.target_net.cuda()

        self.update_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATE*2 + 2))
        self.optimizer = tc.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.record_loss = [None]

    def choose_act(self, x):
        x = tc.unsqueeze(tc.FloatTensor(x), 0)  # .cuda()
        if np.random.uniform() < EPSILON:
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
        self.turbulence_car = '3'
        self.speed_avail = [12, 5, 0, 14, 7, 0, 12, 4, 13, 2, 0, 14, 2, 14, 1]

    def _get_veh_sequence(self, carID):
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
                        veh_list[0] = veh_list[0][veh_list[0].index(carID) + 1:] + veh_list[0][: veh_list[0].index(carID)]
                        for item in veh_list[0]:
                            veh_list[1].append(self._get_interval(carID, item))
                        # print('veh_list: ', veh_list)
                        break
                break
            except ValueError:
                traci.simulationStep()
                print('ValueError')
                continue

        return veh_list

    def _get_interval(self, carID, leader_carID):
        car_lane_index = self.route.index(traci.vehicle.getLaneID(carID))
        leader_car_lane_index = self.route.index(traci.vehicle.getLaneID(leader_carID))
        if leader_car_lane_index < car_lane_index:
            leader_car_lane_index += len(self.route)
        itv = 0.
        for n_lane in range(car_lane_index, leader_car_lane_index):
            itv += traci.lane.getLength(self.route[n_lane % len(self.route)])
        itv -= traci.vehicle.getLanePosition(carID)
        itv += traci.vehicle.getLanePosition(leader_carID)
        if abs(itv) <= 1e-6:
            itv = 1e-6
        if itv < 0:
            for lane in self.route:
                itv += traci.lane.getLength(lane)
        return itv

    def _get_state(self, carID, veh_sq):
        den_former, den_later = self._get_density(carID, veh_sq)
        s = [traci.vehicle.getSpeed(carID),
             veh_sq[1][0],
             # traci.vehicle.getSpeed(carID) - traci.vehicle.getSpeed(veh_sq[0][0]),
             den_former,
             den_later,
             ]
        return s
        # s = []
        # for idx in range(len(self.veh_list)):
        #     s.append(traci.vehicle.getSpeed(self.veh_list[idx]))
        # pos = self._get_relative_pos()
        # pack = zip(s, pos)
        # pack = sorted(pack, key=lambda x: x[1])
        # print('pack: ', pack)
        # s_, pos_ = zip(*pack)
        # return s_ + pos_

    @staticmethod
    def _get_density(carID, veh_sq):
        print('_get_density: ', carID)
        n_former_car = 0
        n_later_car = 0
        former_a = 0
        former_delta_v = 0
        avg_itv_former = 0
        avg_itv_later = 0
        max_interval = 0
        for idx in range(len(veh_sq[0])):
            if veh_sq[1][idx] <= 200:
                n_former_car += 1
                try:
                    avg_itv_former += (veh_sq[1][idx + 1] - veh_sq[1][idx])
                    former_a += traci.vehicle.getAcceleration(veh_sq[0][idx])
                    former_delta_v += (1 + max(1 - idx / 10, 0)) * \
                                      (traci.vehicle.getSpeed(carID) - traci.vehicle.getSpeed(veh_sq[0][idx]))
                except IndexError:
                    avg_itv_former = 0
            elif 428 <= veh_sq[1][idx] < 628:
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
                den_former = min(2 * (traci.vehicle.getSpeed(carID) - traci.vehicle.getSpeed(veh_sq[0][0])) +
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
        print('DENSITY: ', carID, den_former, den_later)
        return den_former, den_later
        # return n_former_car*50, n_later_car * 50

    def _get_reward(self, carID):
        if carID not in traci.simulation.getCollidingVehiclesIDList():  # 3. Calculate the reward
            r = 0
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
                r -= 200
                if self.s[carID][0] > 0:
                    r -= 500
                    r -= self.act[carID] * 150
                    # if act > 0:
                    #     r -= 1000
                    # elif act <= 0:
                    #     r += abs(act) * 400
                # r -= 2. / (s[1] - 4.8) * 1000  # for dangerous headway (new collision punishment)

            # r -= min(400, abs(s[3]) ** 4)
            # r -= min(50, 10 * abs(self.act[carID] - self.act_old[carID]))  # for delta a
            # r += self.s[carID][-1] * 500    # for avg_v
            # r -= abs(self.act[carID]) * 300     # for fuel
            r -= 3 * abs(self.s[carID][2] - self.s[carID][3]) ** 1.2  # for density
        else:
            r = -10000
        return r

    def _get_relative_pos(self):
        relative_pos = []
        for car in self.veh_list:
            car_lane_index = self.route.index(traci.vehicle.getLaneID(car))
            if car_lane_index == 0:
                pos = traci.vehicle.getLanePosition(car)
            elif car_lane_index == 1:
                pos = traci.lane.getLength(self.route[0]) + traci.vehicle.getLanePosition(car)
            relative_pos.append(pos)
        return relative_pos

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

    def run(self):
        global EPSILON
        dqn = DQN()
        traci.simulationStep()

        while True:                                                         # Waiting for all cars drive into route
            traci.simulationStep()
            self.veh_list = list(traci.vehicle.getIDList())
            print('VEH_LIST: ', self.veh_list)
            if len(self.veh_list) != N_CAR:
                for car in self.veh_list:
                    traci.vehicle.setSpeed(car, 2)
                continue
            else:
                for car in self.veh_list:
                    traci.vehicle.setSpeed(car, -1)
                break

        for car in dqn_car:                                                 # Generate initial state
            veh_sq = self._get_veh_sequence(car)
            self.s[car] = self._get_state(car, veh_sq)

        while self.step < EPISODE:                                               # Train start:
            for car in dqn_car:
                self.a[car] = dqn.choose_act(self.s[car])                                 # 1. Choose an action
                self.act[car] = ACT_SPACE[self.a[car]]
                # print('a: ', car, self.act)
                # print 'get lane ID: ', car, traci.vehicle.getLaneID(car)
                # print('v: ', car, self.s[car][0])
                # print 'Lane Position: ', car, traci.vehicle.getLanePosition(car)
                traci.vehicle.setSpeed(car, max(0, self.s[car][0]+0.2*self.act[car]))     # 2. Perform this action

            traci.simulationStep()

            v_sum = 0
            for car in self.veh_list:
                v_sum += traci.vehicle.getSpeed(car) if traci.vehicle.getSpeed(car) >= 0 else 0
            self.avg_v.append(v_sum / len(self.veh_list))
            # self.avg_v.append(traci.vehicle.getSpeed('20') if traci.vehicle.getSpeed('20') >= 0 else 0)

            for car in dqn_car:
                # print('a_target: ', car, traci.vehicle.getAcceleration(car))
                next_tls = traci.vehicle.getNextTLS(car)
                # print('NEXT_TLS: ', car, next_tls)
                veh_sq = self._get_veh_sequence(car)
                r = 0
                """
                    Check if the car is under the control of SUMO (due to TLS...)
                    If so, just update state AND do NOT save data into experience pool, to avoid RL learning wrong data.
                """
                if (round(traci.vehicle.getAcceleration(car), 3) != round(self.act[car] * 0.1, 3)) \
                        and next_tls and next_tls[0][3] == 'r':
                    print(car, 'a_target: ', traci.vehicle.getAcceleration(car))
                    print(car, 'a_act: ', self.act[car] * 0.1)
                    self.reward[car].append(0)
                    self.acc[car].append(traci.vehicle.getAcceleration(car))
                    self.v[car].append(self.s[car][0])
                    self.headway[car].append(self.s[car][1])

                    # interval = self._get_interval(car, veh_sq[0][0])
                    # s_ = [traci.vehicle.getSpeed(car), interval, traci.vehicle.getSpeed(veh_sq[0]),
                    #       traci.vehicle.getSpeed(car) - traci.vehicle.getSpeed(veh_sq[0]),
                    #       traci.vehicle.getSpeed(veh_sq[1]), traci.vehicle.getSpeed(veh_sq[2])]
                    s_ = self._get_state(car, veh_sq)
                    self.s[car] = s_
                    self.act_old[car] = traci.vehicle.getAcceleration(car)
                else:
                    r = self._get_reward(car)
                    # print('INTERVAL: ', car, s[car][1])
                    self.reward[car].append(r)
                    self.acc[car].append(self.act[car])
                    self.v[car].append(self.s[car][0])
                    self.headway[car].append(self.s[car][1])
                    # print('REWARD: ', car, r)

                    # interval = self._get_interval(car, veh_sq[0][0])
                    # s_ = [traci.vehicle.getSpeed(car), interval, traci.vehicle.getSpeed(veh_sq[0]),
                    #       traci.vehicle.getSpeed(car) - traci.vehicle.getSpeed(veh_sq[0]),
                    #       traci.vehicle.getSpeed(veh_sq[1]), traci.vehicle.getSpeed(veh_sq[2])]
                    s_ = self._get_state(car, veh_sq)                            # 4. Get the next state from SUMO

                    dqn.store_transition(self.s[car], self.a[car], r, s_)

                    self.act_old[car] = self.act[car]
                    self.s[car] = s_

                    if self.step >= EPISODE - TEST_TERM:                                  # Bad behaviour count
                        EPSILON = 1.
                        if self.act[car] < -2:
                            self.emer_brake_count[car] += 1
                        if self.s[car][1] < 7:
                            self.dang_headway_count[car] += 1

                    print('---------------------------------\n---REWARD: ',
                          r,
                          '\n---------------------------------')

            if dqn.memory_counter > MEMORY_CAPACITY:                             # 6. Train the eval-net
                dqn.learn()

            self.step += 1
            self.av_step += 1

            v = []
            for car_ in self.veh_list:
                v.append(traci.vehicle.getSpeed(car_)) if traci.vehicle.getSpeed(car_) != -2 ** 30 else v.append(0.)
            pos = self._get_relative_pos()
            self.observer.plot_stop_and_go_wave(N_CAR, pos, v, self.step, 0, 10000, 1000)

            """ Manually change the leader's behaviour to train the dqn-car """
            if self.step % TURBULENCE_TERM == 0 and self.step >= START_TURBULENCE:
                # if self.turbulence_car:
                #     traci.vehicle.setSpeed(self.turbulence_car, -1)
                self.turbulence_car = self.veh_list[25] \
                    if self.turbulence_car != self.veh_list[25] else self.veh_list[20 + np.random.randint(-3, 3)]
                traci.vehicle.setSpeed(self.turbulence_car, 0.5)

            elif self.step % TURBULENCE_TERM == 0.1 * TURBULENCE_TERM and self.step >= START_TURBULENCE:
                traci.vehicle.setSpeed(self.turbulence_car, -1)

            self.observer.plot_var_dyn([self.reward['999'][-1], self.avg_v[-1]], self.step, 30, ['r', 'g'])
            # if self.step % 30 == 0:
            #     plt.figure(0)
            #     plt.cla()
            #     plt.plot(np.linspace(0, self.step, self.step), self.reward['999'], 'b')
            #     plt.figure(1)
            #     plt.plot(np.linspace(0, len(dqn.record_loss), len(dqn.record_loss)), dqn.record_loss, 'g')
            #     plt.ylim(0, 10000)
            #     plt.pause(0.1)
            # print('---------------------------------\n---VALUE_LOSS: ',
            #       dqn.record_loss[-1],
            #       '\n---------------------------------')

        print('-' * 20, '\naverage_v STD: ', np.array(self.avg_v[-TEST_TERM:]).std(), '\n', '-' * 20)
        print('average_v MEAN: ', np.array(self.avg_v[-TEST_TERM:]).mean(), '\n', '-' * 20)
        return self.reward, self.acc, self.v, self.headway, self.emer_brake_count, self.dang_headway_count, \
            self.av_step, dqn.eval_net, dqn.target_net


if __name__ == '__main__':
    sumoBinary = 'sumo-gui'
    sumoCmd = [sumoBinary, '-c', '/home/sandymark/RL-sumo/net.sumocfg', '--collision.action', 'warn']
    traci.start(sumoCmd)

    plt.ion()
    dqn_car = ['999']  # , '998', '997', '996']
    # route1 = ['80_0', '81_0', '82_0', '83_0', '84_0', '85_0']
    route1 = ['edge2_0', 'edge3_0']
    trainer = Train()
    re, a, vel, hdway, e_b, d_h, t, eval_net, target_net = trainer.run()
    plt.ioff()
    plt.show()
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
        tc.save(eval_net, '/home/sandymark/RL-sumo/net/evalnet' + str(num) + '.torch')
        tc.save(target_net, '/home/sandymark/RL-sumo/net/targetnet' + str(num) + '.torch')

    traci.close()
