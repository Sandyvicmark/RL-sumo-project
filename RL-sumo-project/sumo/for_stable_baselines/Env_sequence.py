# import os
# import sys
# import gym
# import math
# import numpy as np
# from Tools import SumoSDK
# from Tools.Statistics import Observer
# from RLlib import DDPG
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")
# import traci
#
# SIM_STEP_LENGTH = 0.1
# N_CAR = 25
# ROUTE = ['edge290_0', 'edge390_0']
# ROUTE_LENGTH = 283.
# DQN_CAR = ['999']
#
# TURBULENCE_TERM = 20
# START_TURBULENCE = 2500
#
#
# class CustomEnv(gym.Env):
#     metadata = {'render.modes': ['human']}
#
#     def __init__(self):
#         super(CustomEnv, self).__init__()
#         self.action_space = gym.spaces.Box(low=-2, high=2, shape=(1,), dtype=np.float)
#         # self.action_space = gym.spaces.Discrete(50)
#         self.observation_space = gym.spaces.Box(low=np.zeros((4,)),
#                                                 high=np.ones((4,)),
#                                                 shape=(4,),
#                                                 dtype=np.float)
#         # self.observation_space = gym.spaces.Box(low=np.zeros((48,)),
#         #                                         high=np.hstack((15 * np.ones((24,)), 650 * np.ones((24,)))),
#         #                                         shape=(48,),
#         #                                         dtype=np.float)
#
#         self.route = ROUTE
#         self.dqn_car = DQN_CAR
#         self.observer = Observer()
#         # self.reset_pos_x = [100.0, 98.76883405951378, 95.10565162951535, 89.10065241883679, 80.90169943749474,
#         #                     70.71067811865476, 58.778525229247315, 45.39904997395468, 30.901699437494745,
#         #                     15.643446504023093, 6.123233995736766e-15, -15.643446504023103, -30.901699437494734,
#         #                     -45.39904997395467, -58.7785252292473, -70.71067811865474, -80.90169943749473,
#         #                     -89.10065241883677, -95.10565162951535, -98.76883405951376, -100.0, -98.76883405951378,
#         #                     -95.10565162951535, -89.10065241883682, -80.90169943749476, -70.71067811865477,
#         #                     -58.77852522924732, -45.39904997395469, -30.901699437494756, -15.643446504023103,
#         #                     -1.8369701987210297e-14, 15.643446504023068, 30.901699437494724, 45.39904997395466,
#         #                     58.77852522924729, 70.71067811865474, 80.90169943749473, 89.10065241883677,
#         #                     95.10565162951535, 98.76883405951376]
#         # self.reset_pos_y = [200.0, 215.64344650402307, 230.90169943749473, 245.39904997395467, 258.7785252292473,
#         #                     270.71067811865476, 280.90169943749476, 289.1006524188368, 295.10565162951536,
#         #                     298.7688340595138, 300.0, 298.7688340595138, 295.10565162951536, 289.1006524188368,
#         #                     280.90169943749476, 270.71067811865476, 258.7785252292473, 245.39904997395467,
#         #                     230.90169943749476, 215.6434465040231, 200.0, 184.35655349597693, 169.09830056250522,
#         #                     154.60095002604538, 141.2214747707527, 129.28932188134524, 119.09830056250527,
#         #                     110.89934758116323, 104.89434837048465, 101.23116594048624, 100.0, 101.23116594048622,
#         #                     104.89434837048464, 110.89934758116321, 119.09830056250524, 129.28932188134524,
#         #                     141.22147477075265, 154.6009500260453, 169.09830056250524, 184.3565534959769]
#         self.reset_pos_x = [45.0, 44.276831486938335, 42.13056917878817, 38.630195712083975, 33.88821597016249, 28.05704108364301, 21.324089811284942, 13.905764746872634, 6.040496961794498, -2.018917365773159, -10.013442028034145, -17.68612642442656, -24.79036416534461, -31.097819204408903, -36.40576474687263, -40.54359905560886, -43.378328731313395, -44.818843229785756, -44.818843229785756, -43.3783287313134, -40.54359905560886, -36.405764746872634, -31.097819204408914, -24.79036416534462, -17.68612642442657, -10.013442028034156, -2.01891736577318, 6.040496961794487, 13.905764746872626, 21.324089811284935, 28.057041083643, 33.88821597016249, 38.63019571208397, 42.13056917878817, 44.276831486938335]
#
#         self.reset_pos_y = [200.0, 208.03506026593865, 215.81186708366042, 223.08046748326578, 229.60724266728707, 235.18241671106134, 239.6267989335532, 242.7975432332819, 244.59273927955707, 244.95468799435918, 243.87175604818208, 241.37874976481527, 237.55579641745862, 232.52576887223262, 226.4503363531613, 219.52476826029013, 211.9716580505004, 204.0337689006545, 195.96623109934552, 188.02834194949963, 180.4752317397099, 173.54966364683872, 167.47423112776738, 162.44420358254138, 158.62125023518473, 156.12824395181795, 155.04531200564085, 155.40726072044293, 157.2024567667181, 160.37320106644677, 164.81758328893866, 170.39275733271293, 176.91953251673422, 184.18813291633955, 191.96493973406135]
#
#         # self.turbulence_car = '25'
#         self.veh_list = []
#         self.s = {}
#         self.a = {}
#         self.done = {}
#         self.veh_sq = None
#         self.act = {}
#         self.act_old = {}
#         self.reward = {}
#         self.log_prob = {}
#         self.acc = {}
#         self.v = {}
#         self.avg_v = [0.]
#         self.headway = {}
#         self.fuel = {}
#         self.emer_brake_count = {}
#         self.dang_headway_count = {}
#         self.ok_flag = False
#         self.last_turbulence_car = None
#         for item in DQN_CAR:
#             self.act_old[item] = 0.
#             self.reward[item] = []
#             self.log_prob[item] = []
#             self.acc[item] = []
#             self.v[item] = []
#             self.headway[item] = []
#             self.emer_brake_count[item] = 0
#             self.dang_headway_count[item] = 0
#             self.done[item] = 0
#         self.i_episode = 0
#         self.n_step = 0
#         self.av_step = 0
#
#         sumo_binary = 'sumo-gui'
#         sumo_cmd = [sumo_binary, '-c', '/Users/sandymark/RL-sumo/net.sumocfg', '--collision.action', 'warn']
#         traci.start(sumo_cmd)
#
#         self.veh_list = SumoSDK.wait_all_vehicles(N_CAR)
#
#     def reset_one(self, car):
#         x = self.reset_pos_x
#         y = self.reset_pos_y
#         reset_idx = np.random.choice(len(x))
#         if reset_idx < (len(x) // 2):
#             edge = 'edge290'
#         else:
#             edge = 'edge390'
#         traci.vehicle.moveToXY(car, edge, 0, x[reset_idx], y[reset_idx])
#         traci.vehicle.setSpeed(car, 0)
#         traci.simulationStep()
#         traci.vehicle.setSpeed(car, -1)
#
#     def reset(self, return_state=True):
#         for _ in DQN_CAR:
#             self.done[_] = 0
#         x = self.reset_pos_x
#         y = self.reset_pos_y
#         reset_idx = list(np.random.choice(len(x), (N_CAR,), replace=False))
#         veh_list = sorted(self.veh_list, key=lambda x: int(x))
#         # traci.vehicle.moveToXY('0', 'edge2', 0, self.reset_pos_x[0], self.reset_pos_y[0])
#         # traci.vehicle.moveToXY('999', 'edge2', 0, self.reset_pos_x[1], self.reset_pos_y[1])
#         for idx, item in zip(range(len(reset_idx)), reset_idx):
#             if item < (len(x) // 2):
#                 edge = 'edge290'
#             else:
#                 edge = 'edge390'
#             traci.vehicle.moveToXY(veh_list[idx], edge, 0, x[item], y[item])
#             traci.vehicle.setSpeed(veh_list[idx], 0)
#         # for idx in range(len(veh_list)):
#         #
#         #     if idx < 20:
#         #         traci.vehicle.moveToXY(veh_list[idx], 'edge2', 1, self.reset_pos_x[idx], self.reset_pos_y[idx])
#         #     else:
#         #         traci.vehicle.moveToXY(veh_list[idx], 'edge3', 1, self.reset_pos_x[idx], self.reset_pos_y[idx])
#         #     traci.vehicle.setSpeed(veh_list[idx], 1)
#         traci.simulationStep()
#         for car_ in veh_list:
#             traci.vehicle.setSpeed(car_, -1)
#
#         if return_state:
#             veh_sq = self._get_veh_sequence()               # Generate initial state
#             for car_ in DQN_CAR:
#                 car_idx = veh_sq[0].index(car_)
#                 veh_sq_ = [veh_sq[0][car_idx + 1:] + veh_sq[0][: car_idx],
#                            veh_sq[1][car_idx + 1:] + veh_sq[1][: car_idx]]
#                 veh_sq_[1] = self._get_interval(veh_sq[1][car_idx], veh_sq_)
#                 self.s[car_] = self._get_state(car_, veh_sq_)
#             # get centralized state
#             # s = []
#             # for _ in DQN_CAR:
#             #     s += self.s[_]
#             return self.s['999']
#
#     def step(self, action: dict):
#         # Take action
#         # action = np.clip(action, -3., 2.)
#         for _ in action.keys():
#             traci.vehicle.setSpeed(_, max(0, traci.vehicle.getSpeed(_) + SIM_STEP_LENGTH * action[_]))
#         # print(action)
#             self.act[_] = action[_]
#         # self.act['999'] = action / 10 - 3
#         traci.simulationStep()
#
#         # Get reward
#         reward = {}
#         for _ in DQN_CAR:
#             reward[_] = self._get_reward_test(_)
#         cent_reward = 0
#         for value in reward.values():
#             cent_reward += value
#         # print(reward)
#
#         # Get next state
#         veh_sq = self._get_veh_sequence()
#         for car_ in DQN_CAR:
#             car_idx = veh_sq[0].index(car_)
#             veh_sq_ = [veh_sq[0][car_idx + 1:] + veh_sq[0][: car_idx],
#                        veh_sq[1][car_idx + 1:] + veh_sq[1][: car_idx]]
#             veh_sq_[1] = self._get_interval(veh_sq[1][car_idx], veh_sq_)
#             s_ = self._get_state(car_, veh_sq_)            # Only take 999's state for single-car test
#             self.s[car_] = s_
#         # get centralized state
#         # s_ = []
#         # for _ in DQN_CAR:
#         #     s_ += self.s[_]
#
#         # Reset if collision occurred
#         collision_list = traci.simulation.getCollidingVehiclesIDList()
#         for _ in DQN_CAR:
#             if _ in collision_list:
#                 # self.reset(return_state=False)
#                 self.done[_] = 1
#             else:
#                 self.done[_] = 0
#
#         self.avg_v.append(traci.vehicle.getSpeed('5') if traci.vehicle.getSpeed('5') >= 0 else 0)
#
#         """ Manually change the leader's behaviour to train the dqn-car """
#         if self.n_step % TURBULENCE_TERM == 0 and self.n_step >= START_TURBULENCE:
#             # if self.step == 300:
#             # if self.turbulence_car:
#             #     traci.vehicle.setSpeed(self.turbulence_car, -1)
#             # self.turbulence_car = self.veh_list[25] \
#             #     if self.turbulence_car != self.veh_list[25] else self.veh_list[20 + np.random.randint(-3, 3)]
#             # turbulence_car = self.turbulence_car[np.random.choice(len(self.turbulence_car))]
#             self.last_turbulence_car = str(np.random.randint(N_CAR - 2))
#             traci.vehicle.setSpeed(self.last_turbulence_car, np.random.random() * 5 + 1)  # 0.01 + np.random.random() / 2)
#
#         elif self.n_step % TURBULENCE_TERM == 10 and self.n_step > START_TURBULENCE:
#             # elif self.step == 320 and self.step >= START_TURBULENCE:
#             traci.vehicle.setSpeed(self.last_turbulence_car, -1)
#
#         self.n_step += 1
#         # if self.n_step == 0:
#         #     traci.vehicletype.setMinGap('car', 10)
#         if self.n_step > 5000:
#             self.observer.plot_var_dyn([reward['999'], self.act['999']], self.n_step, 300, [0, 0], 1, ['b', 'r'])
#         return self.s, reward, self.done, {}
#
#     def render(self, mode='human'):
#         pass
#
#     def _get_veh_sequence(self):
#         """ Generate a list, storing the car sequence before dqn-car,
#             in which the closest former car is the first element.
#
#             Note that: ABSOLUTE POSITION will be stored in veh_list[1]"""
#         while True:
#             veh_list = [[], []]
#             try:
#                 while True:
#                     for lane in self.route:
#                         veh_list[0] += traci.lane.getLastStepVehicleIDs(lane)
#                     if len(veh_list[0]) != N_CAR:
#                         traci.simulationStep()
#                         continue
#                     else:
#                         abs_pos = self._get_absolute_pos(veh_list[0])
#                         veh_list[1] = abs_pos
#                         # for item in veh_list[0]:
#                         #     veh_list[1].append(self._get_interval(carID, item))
#                         # print('veh_list: ', veh_list)
#                         break
#                 break
#             except ValueError:
#                 traci.simulationStep()
#                 print('ValueError')
#                 continue
#
#         return veh_list
#
#     def _get_absolute_pos(self, veh_sq):
#         abs_pos = []
#         for car_ in veh_sq:
#             car_lane_index = self.route.index(traci.vehicle.getLaneID(car_))
#             if car_lane_index == 0:
#                 pos = traci.vehicle.getLanePosition(car_)
#             elif car_lane_index == 1:
#                 pos = traci.lane.getLength(self.route[0]) + traci.vehicle.getLanePosition(car_)
#             abs_pos.append(pos)
#         return abs_pos
#
#     # def _get_cent_state(self, veh_sq_):
#     #     s = []
#     #     for _ in DQN_CAR:
#     #         den_former, den_later, vhf, vhl, nf, nl = self._get_density(veh_sq_)
#     #         s = [traci.vehicle.getSpeed(car),
#     #              veh_sq_[1][0],
#     #              (ROUTE_LENGTH - veh_sq_[1][-1]),
#     #              (traci.vehicle.getSpeed(car) - traci.vehicle.getSpeed(veh_sq_[0][0])),
#     #              (den_former - den_later)
#     #              ]
#
#     def _get_state(self, car, veh_sq_):
#         den_former, den_later, vhf, vhl, nf, nl = self._get_density(veh_sq_)
#         s = [traci.vehicle.getSpeed(car),
#              veh_sq_[1][0],
#              (ROUTE_LENGTH - veh_sq_[1][-1]),
#              # (traci.vehicle.getSpeed(car) - traci.vehicle.getSpeed(veh_sq_[0][0])),
#              (den_former - den_later)
#              ]
#         s_norm = [s[0] / 15,
#                   s[1] / 110,
#                   s[2] / 110,
#                   # s[3] / 20 + 0.5,
#                   s[3] / 2000 + 0.5]
#         # print('STATE: ', s_norm)
#         return s_norm
#
#     @staticmethod
#     def _get_density(veh_sq):
#         look_range = 100         # default: 200
#         former_car_list = []
#         later_car_list = []
#         for idx in range(len(veh_sq[0])):
#             if 0 <= veh_sq[1][idx] < look_range:
#                 former_car_list.append(veh_sq[0][idx])
#             elif ROUTE_LENGTH - look_range < veh_sq[1][idx] < ROUTE_LENGTH:
#                 later_car_list.append(veh_sq[0][idx])
#
#         n_former_car = len(former_car_list)
#         n_later_car = len(later_car_list)
#
#         v_h_for = 1e-6
#         v_h_lat = 1e-6
#         sum_d_for = 0
#         sum_d_lat = 0
#         w_v_f = 0
#         w_v_l = 0
#         w_p_f = 0
#         w_p_l = 0
#         for idx, car_ in zip(range(len(former_car_list)), former_car_list):
#             # v_h_for += 1 / max(0.1, traci.vehicle.getSpeed(car_))  # Harmonic Sum
#             # v_h_for += max(0.1, traci.vehicle.getSpeed(car_))  # Arithmetic Sum
#             v_h_for += (1 - math.pow(veh_sq[1][idx] / 400, 1)) / max(0.1, traci.vehicle.getSpeed(
#                 car_))  # Weighted Harmonic Sum
#             w_v_f += (1 - math.pow(veh_sq[1][idx] / 400, 1))
#             # sum_d_for += (1 - traci.vehicle.getAcceleration(car_) / 3) / veh_sq[1][idx]
#             # w_p_f += (1 - traci.vehicle.getAcceleration(car_) / 3)
#             if idx == 0:
#                 sum_d_for += 1 / veh_sq[1][0]
#                 # w_p_f += 200
#                 w_p_f += 1
#             else:
#                 sum_d_for += (1 - math.pow(veh_sq[1][idx - 1] / 400, 1)) / max(0.1, veh_sq[1][idx] - veh_sq[1][idx - 1])
#                 # w_p_f += (200 - veh_sq[1][idx - 1])
#                 w_p_f += (1 - math.pow(veh_sq[1][idx - 1] / 400, 1))
#         # v_h_for = max(1 / v_h_for, 0.1)  # Harmonic Mean
#         # v_h_for = max(v_h_for / n_former_car, 0.1)  # Arithmetic Mean
#         v_h_for = max(w_v_f / v_h_for, 0.1)  # Weighted Harmonic Mean
#         for idx, car_ in zip(range(len(later_car_list)), later_car_list):
#             # v_h_lat += 1 / max(0.1, traci.vehicle.getSpeed(car_))
#             # v_h_lat += max(0.1, traci.vehicle.getSpeed(car_))
#             v_h_lat += (1 - math.pow((ROUTE_LENGTH - veh_sq[1][idx - n_later_car]) / 400, 1)) / max(0.1,
#                                                                                            traci.vehicle.getSpeed(car_))
#             w_v_l += (1 - math.pow((ROUTE_LENGTH - veh_sq[1][idx - n_later_car]) / 400, 1))
#             # sum_d_lat += (1 - traci.vehicle.getAcceleration(car_) / 3) / (veh_sq[1][idx - n_later_car + 1] - veh_sq[1][idx - n_later_car])
#             # w_p_l += (1 - traci.vehicle.getAcceleration(car_) / 3)
#             if idx == n_later_car - 1:
#                 sum_d_lat += 1 / (ROUTE_LENGTH - veh_sq[1][-1])
#                 # w_p_l += 200
#                 w_p_l += 1
#             else:
#                 sum_d_lat += (1 - math.pow((ROUTE_LENGTH - veh_sq[1][idx - n_later_car + 1]) / 400, 1)) / max(0.1, veh_sq[1][
#                     idx - n_later_car + 1] - veh_sq[1][idx - n_later_car])
#                 # w_p_l += (200 - (ROUTE_LENGTH - veh_sq[1][idx - n_later_car + 1]))
#                 w_p_l += (1 - math.pow((ROUTE_LENGTH - veh_sq[1][idx - n_later_car + 1]) / 400, 1))
#         # v_h_lat = max(1 / v_h_lat, 0.1)
#         # v_h_lat = max(v_h_lat / n_later_car, 0.1)
#         v_h_lat = max(w_v_l / v_h_lat, 0.1)
#         if n_former_car:
#             # den_former = n_former_car + 5 * (1 - math.pow(veh_sq[1][0] / 600, 1)) / v_h_for + 1000 / max(0.1, veh_sq[1][0])
#             den_former = 300 * sum_d_for / w_p_f + 15 / v_h_for + 500 / max(0.1, veh_sq[1][0])
#         else:
#             den_former = 0
#         if n_later_car:
#             # den_later = n_later_car + 5 * (1 - math.pow((629 - veh_sq[1][-1]) / 600, 1)) / v_h_lat + 1000 / max(0.1, 629 - veh_sq[1][-1])
#             den_later = 300 * sum_d_lat / w_p_l + 15 / v_h_lat + 500 / max(0.1, (ROUTE_LENGTH - veh_sq[1][-1]))
#         else:
#             den_later = 0
#
#         # print('den_former: ', den_former, 'sdf: ', sum_d_for, 'wpf: ', w_p_f, 'vh: ', v_h_for, 'itv: ', veh_sq[1][0])
#         # print('den_later: ', den_later, 'sdll: ', sum_d_lat, 'wpl: ', w_p_l, 'vh: ', v_h_lat, 'itv: ', ROUTE_LENGTH - veh_sq[1][-1])
#         return min(1000., den_former * 2.5), min(1000., den_later * 2.5), v_h_for, v_h_lat, n_former_car, n_later_car
#         # return n_former_car*50, n_later_car * 50
#
#     def _get_reward(self, carID):
#         r = 0  # 3. Calculate the reward
#         # s_star = 5 + 3 + max(0, self.s[carID][0] * T + self.s[carID][0] * (self.s[carID][2]) / (2 * math.sqrt(2. * 3.)))
#         # a_star = min(2, max(-3, 2. * (1 - (self.s[carID][0] / 15.) ** 4 - (s_star / self.s[carID][1]) ** 2)))
#         # print('a_star: ', carID, a_star)
#         # r = -150 * abs(self.act[carID] - a_star)  # for a
#
#         # if ((traci.vehicle.getAcceleration(veh_sq[0]) <= -2 and s[car][1] <= 20) or
#         #     (s[car][3] >= 5 and s[car][1] <= (s[car][0]**2 - s[car][2]**2) / 2*2.5 + 3)) and act[car] > -2:
#         #     r -= 1000
#         #
#         # if s[car][1] <= 7 and act[car] >= -2:
#         #     r -= 1000
#         # elif s[car][1] <= 7 and act[car] < -2:
#         #     r += 500
#
#         # if (traci.vehicle.getAcceleration(veh_sequence[0][0]) <= -2 and self.s[carID][1] <= 20) or \
#         #         (self.s[carID][3] >= 5 and self.s[carID][1] <= (self.s[carID][0] ** 2 - traci.vehicle.getSpeed(veh_sequence[0][0]) ** 2) / 2 * 2.5 + 3):  # or \
#         #     # s[1] <= 7:
#         #     if self.act[carID] > -2:
#         #         r -= 2000
#         #     # else:
#         #     #     r += 500                    # for essential emergency break
#
#         if self.s[carID][1] * 40 <= 7:
#             r -= 500
#             if self.s[carID][0] > 0:
#                 r -= 500
#                 r -= self.act[carID] * 150
#                 # if act > 0:
#                 #     r -= 1000
#                 # elif act <= 0:
#                 #     r += abs(act) * 400
#             # r -= 15 / max(0.1, self.s[carID][0]) * 100  # for dangerous headway (new collision punishment)
#
#         # r -= min(400, abs(s[3]) ** 4)
#         r -= 100 * abs(self.act[carID] - self.act_old[carID])  # for delta a
#         # r += self.s[carID][-1] * 500    # for avg_v
#         # r -= abs(self.act[carID]) * 300     # for fuel
#         r -= 30 * abs(self.s[carID][4] * 4000 - 2000)  # for density
#         r -= 80 * abs(self.act[carID]) ** 2
#         # if carID in traci.simulation.getCollidingVehiclesIDList():
#         if self.s[carID][1] * 40 <= 5:
#             r -= 10000
#         return max(-20000, r) / 20000 + 1
#
#     def _get_reward_test(self, car):
#         r = 0
#         r -= abs(self.s[car][3] - 0.5)
#         # v = np.zeros((N_CAR,))
#         # for car, idx in zip(self.veh_list, range(len(self.veh_list))):
#         #     v[idx] = traci.vehicle.getSpeed(car)
#         # r += v.mean() / 5
#         # if 5 < self.s[car][1] * 90 <= 7:
#         #     r = -1.5
#         # elif self.s[car][1] * 90 <= 5:
#         #     r = -2
#         return r
#
#     @staticmethod
#     def _get_interval(pos_cur_car, veh_sq):
#         veh_sq[1] -= pos_cur_car * np.ones((len(veh_sq[1])))
#         for idx in range(len(veh_sq[1])):
#             if veh_sq[1][idx] < 0:
#                 veh_sq[1][idx] += ROUTE_LENGTH
#         return list(veh_sq[1])

import gym
import math
import numpy as np
from Tools import SumoSDK
from Tools.Statistics import Observer
from RLlib import DDPG
import traci

SIM_STEP_LENGTH = 0.1
N_CAR = 25
ROUTE = ['edge290_0', 'edge390_0']
ROUTE_LENGTH = 283.
DQN_CAR = ['0']

TURBULENCE_TERM = 400
START_TURBULENCE = 10000


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-2, high=2, shape=(1,), dtype=np.float)
        # self.action_space = gym.spaces.Discrete(50)
        self.observation_space = gym.spaces.Box(low=np.zeros((4,)),
                                                high=np.ones((4,)),
                                                shape=(4,),
                                                dtype=np.float)
        # self.observation_space = gym.spaces.Box(low=np.zeros((48,)),
        #                                         high=np.hstack((15 * np.ones((24,)), 650 * np.ones((24,)))),
        #                                         shape=(48,),
        #                                         dtype=np.float)

        self.route = ROUTE
        self.dqn_car = DQN_CAR
        self.observer = Observer()
        # self.reset_pos_x = [100.0, 98.76883405951378, 95.10565162951535, 89.10065241883679, 80.90169943749474,
        #                     70.71067811865476, 58.778525229247315, 45.39904997395468, 30.901699437494745,
        #                     15.643446504023093, 6.123233995736766e-15, -15.643446504023103, -30.901699437494734,
        #                     -45.39904997395467, -58.7785252292473, -70.71067811865474, -80.90169943749473,
        #                     -89.10065241883677, -95.10565162951535, -98.76883405951376, -100.0, -98.76883405951378,
        #                     -95.10565162951535, -89.10065241883682, -80.90169943749476, -70.71067811865477,
        #                     -58.77852522924732, -45.39904997395469, -30.901699437494756, -15.643446504023103,
        #                     -1.8369701987210297e-14, 15.643446504023068, 30.901699437494724, 45.39904997395466,
        #                     58.77852522924729, 70.71067811865474, 80.90169943749473, 89.10065241883677,
        #                     95.10565162951535, 98.76883405951376]
        # self.reset_pos_y = [200.0, 215.64344650402307, 230.90169943749473, 245.39904997395467, 258.7785252292473,
        #                     270.71067811865476, 280.90169943749476, 289.1006524188368, 295.10565162951536,
        #                     298.7688340595138, 300.0, 298.7688340595138, 295.10565162951536, 289.1006524188368,
        #                     280.90169943749476, 270.71067811865476, 258.7785252292473, 245.39904997395467,
        #                     230.90169943749476, 215.6434465040231, 200.0, 184.35655349597693, 169.09830056250522,
        #                     154.60095002604538, 141.2214747707527, 129.28932188134524, 119.09830056250527,
        #                     110.89934758116323, 104.89434837048465, 101.23116594048624, 100.0, 101.23116594048622,
        #                     104.89434837048464, 110.89934758116321, 119.09830056250524, 129.28932188134524,
        #                     141.22147477075265, 154.6009500260453, 169.09830056250524, 184.3565534959769]
        self.reset_pos_x = [45.0, 44.276831486938335, 42.13056917878817, 38.630195712083975, 33.88821597016249, 28.05704108364301, 21.324089811284942, 13.905764746872634, 6.040496961794498, -2.018917365773159, -10.013442028034145, -17.68612642442656, -24.79036416534461, -31.097819204408903, -36.40576474687263, -40.54359905560886, -43.378328731313395, -44.818843229785756, -44.818843229785756, -43.3783287313134, -40.54359905560886, -36.405764746872634, -31.097819204408914, -24.79036416534462, -17.68612642442657, -10.013442028034156, -2.01891736577318, 6.040496961794487, 13.905764746872626, 21.324089811284935, 28.057041083643, 33.88821597016249, 38.63019571208397, 42.13056917878817, 44.276831486938335]

        self.reset_pos_y = [200.0, 208.03506026593865, 215.81186708366042, 223.08046748326578, 229.60724266728707, 235.18241671106134, 239.6267989335532, 242.7975432332819, 244.59273927955707, 244.95468799435918, 243.87175604818208, 241.37874976481527, 237.55579641745862, 232.52576887223262, 226.4503363531613, 219.52476826029013, 211.9716580505004, 204.0337689006545, 195.96623109934552, 188.02834194949963, 180.4752317397099, 173.54966364683872, 167.47423112776738, 162.44420358254138, 158.62125023518473, 156.12824395181795, 155.04531200564085, 155.40726072044293, 157.2024567667181, 160.37320106644677, 164.81758328893866, 170.39275733271293, 176.91953251673422, 184.18813291633955, 191.96493973406135]

        # self.turbulence_car = '25'
        self.veh_list = []
        self.s = {}
        self.a = {}
        self.done = 0
        self.veh_sq = None
        self.act = {}
        self.act_old = {}
        self.reward = {}
        self.log_prob = {}
        self.acc = {}
        self.v = {}
        self.avg_v = [0.]
        self.headway = {}
        self.fuel = {}
        self.emer_brake_count = {}
        self.dang_headway_count = {}
        self.ok_flag = False
        self.last_turbulence_car = None
        self.past_roue = [0.]
        for item in DQN_CAR:
            self.act_old[item] = 0.
            self.reward[item] = []
            self.log_prob[item] = []
            self.acc[item] = []
            self.v[item] = []
            self.headway[item] = []
            self.emer_brake_count[item] = 0
            self.dang_headway_count[item] = 0
        self.i_episode = 0
        self.n_step = 0
        self.av_step = 0

        sumo_binary = 'sumo-gui'
        sumo_cmd = [sumo_binary, '-c', '/Users/sandymark/RL-sumo/net.sumocfg', '--collision.action', 'warn']
        traci.start(sumo_cmd)

        self.veh_list = SumoSDK.wait_all_vehicles(N_CAR)

    def reset_one(self, car):
        x = self.reset_pos_x
        y = self.reset_pos_y
        reset_idx = np.random.choice(len(x))
        if reset_idx < (len(x) // 2):
            edge = 'edge290'
        else:
            edge = 'edge390'
        traci.vehicle.moveToXY(car, edge, 0, x[reset_idx], y[reset_idx])
        traci.vehicle.setSpeed(car, 0)
        traci.simulationStep()
        traci.vehicle.setSpeed(car, -1)

    def reset(self, return_state=True):
        self.done = 0
        # self.past_roue = [0.]
        x = self.reset_pos_x
        y = self.reset_pos_y
        reset_idx = list(np.random.choice(len(x), (N_CAR,), replace=False))
        veh_list = sorted(self.veh_list, key=lambda x: int(x))
        # traci.vehicle.moveToXY('0', 'edge2', 0, self.reset_pos_x[0], self.reset_pos_y[0])
        # traci.vehicle.moveToXY('999', 'edge2', 0, self.reset_pos_x[1], self.reset_pos_y[1])
        for idx, item in zip(range(len(reset_idx)), reset_idx):
            if item < (len(x) // 2):
                edge = 'edge290'
            else:
                edge = 'edge390'
            traci.vehicle.moveToXY(veh_list[idx], edge, 0, x[item], y[item])
            traci.vehicle.setSpeed(veh_list[idx], 0)
        # for idx in reversed(range(N_CAR)):
        #     if idx < (len(x) // 2):
        #         traci.vehicle.moveToXY(veh_list[idx], 'edge290', 1, self.reset_pos_x[idx], self.reset_pos_y[idx])
        #     else:
        #         traci.vehicle.moveToXY(veh_list[idx], 'edge390', 1, self.reset_pos_x[idx], self.reset_pos_y[idx])
        #     traci.vehicle.setSpeed(veh_list[idx], 0)
        # for idx in range(len(veh_list)):
        #
        #     if idx < 20:
        #         traci.vehicle.moveToXY(veh_list[idx], 'edge2', 1, self.reset_pos_x[idx], self.reset_pos_y[idx])
        #     else:
        #         traci.vehicle.moveToXY(veh_list[idx], 'edge3', 1, self.reset_pos_x[idx], self.reset_pos_y[idx])
        #     traci.vehicle.setSpeed(veh_list[idx], 1)
        traci.simulationStep()
        for car_ in veh_list:
            traci.vehicle.setSpeed(car_, -1)

        if return_state:
            veh_sq = self._get_veh_sequence()               # Generate initial state
            for car_ in DQN_CAR:
                car_idx = veh_sq[0].index(car_)
                veh_sq_ = [veh_sq[0][car_idx + 1:] + veh_sq[0][: car_idx],
                           veh_sq[1][car_idx + 1:] + veh_sq[1][: car_idx]]
                veh_sq_[1] = self._get_interval(veh_sq[1][car_idx], veh_sq_)
                self.s[car_] = self._get_state(car_, veh_sq_)
            # get centralized state
            # s = []
            # for _ in DQN_CAR:
            #     s += self.s[_]
            return self.s['999']

    def step(self, action):
        # Take action
        # action = np.clip(action, -3., 2.)
        traci.vehicle.setSpeed('999', max(0, traci.vehicle.getSpeed('999') + SIM_STEP_LENGTH * action[0]))
        # print(action)
        self.act['999'] = action[0]
        # self.act['999'] = action / 10 - 3
        traci.simulationStep()

        # Get reward
        reward = 0
        # reward = self._get_reward_test('999')
        reward = traci.vehicle.getSpeed('999')
        # cent_reward = 0
        # for value in reward.values():
        #     cent_reward += value
        # # print(reward)

        # Get next state
        veh_sq = self._get_veh_sequence()
        for car_ in DQN_CAR:
            car_idx = veh_sq[0].index(car_)
            veh_sq_ = [veh_sq[0][car_idx + 1:] + veh_sq[0][: car_idx],
                       veh_sq[1][car_idx + 1:] + veh_sq[1][: car_idx]]
            veh_sq_[1] = self._get_interval(veh_sq[1][car_idx], veh_sq_)
            s_ = self._get_state(car_, veh_sq_)            # Only take 999's state for single-car test
            self.s[car_] = s_

        self.past_roue.append(self.s['999'][3])
        if len(self.past_roue) > 20:
            self.past_roue.pop(0)
        # get centralized state
        # s_ = []
        # for _ in DQN_CAR:
        #     s_ += self.s[_]

        # Reset if collision occurred
        collision_list = traci.simulation.getCollidingVehiclesIDList()
        if collision_list:
            # self.reset(return_state=False)
            self.done = 1
        else:
            self.done = 0

        self.avg_v.append(traci.vehicle.getSpeed('5') if traci.vehicle.getSpeed('5') >= 0 else 0)

        """ Manually change the leader's behaviour to train the dqn-car """
        if self.n_step % TURBULENCE_TERM == 0 and self.n_step >= START_TURBULENCE:
            # if self.step == 300:
            # if self.turbulence_car:
            #     traci.vehicle.setSpeed(self.turbulence_car, -1)
            # self.turbulence_car = self.veh_list[25] \
            #     if self.turbulence_car != self.veh_list[25] else self.veh_list[20 + np.random.randint(-3, 3)]
            # turbulence_car = self.turbulence_car[np.random.choice(len(self.turbulence_car))]
            self.last_turbulence_car = str(np.random.randint(N_CAR - 2))
            traci.vehicle.setSpeed(self.last_turbulence_car, np.random.random() + 0.1)  # 0.01 + np.random.random() / 2)

        elif self.n_step % TURBULENCE_TERM == 100 and self.n_step > START_TURBULENCE:
            # elif self.step == 320 and self.step >= START_TURBULENCE:
            traci.vehicle.setSpeed(self.last_turbulence_car, -1)

        self.n_step += 1
        # if self.n_step == 0:
        #     traci.vehicletype.setMinGap('car', 10)
        if self.n_step > 1000:
            self.observer.plot_var_dyn([reward, self.act['999']], self.n_step, 100, [0, 0], 1, ['b', 'r'])
        return self.s['999'], reward, self.done, {}

    def render(self, mode='human'):
        pass

    def _get_veh_sequence(self):
        """ Generate a list, storing the car sequence before dqn-car,
            in which the closest former car is the first element.

            Note that: ABSOLUTE POSITION will be stored in veh_list[1]"""
        while True:
            veh_list = [[], []]
            try:
                while True:
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

    def _get_absolute_pos(self, veh_sq):
        abs_pos = []
        for car_ in veh_sq:
            car_lane_index = self.route.index(traci.vehicle.getLaneID(car_))
            if car_lane_index == 0:
                pos = traci.vehicle.getLanePosition(car_)
            elif car_lane_index == 1:
                pos = traci.lane.getLength(self.route[0]) + traci.vehicle.getLanePosition(car_)
            abs_pos.append(pos)
        return abs_pos

    # def _get_cent_state(self, veh_sq_):
    #     s = []
    #     for _ in DQN_CAR:
    #         den_former, den_later, vhf, vhl, nf, nl = self._get_density(veh_sq_)
    #         s = [traci.vehicle.getSpeed(car),
    #              veh_sq_[1][0],
    #              (ROUTE_LENGTH - veh_sq_[1][-1]),
    #              (traci.vehicle.getSpeed(car) - traci.vehicle.getSpeed(veh_sq_[0][0])),
    #              (den_former - den_later)
    #              ]

    def _get_state(self, car, veh_sq_):
        den_former, den_later, vhf, vhl, nf, nl = self._get_density(veh_sq_)
        s = [traci.vehicle.getSpeed(car),
             veh_sq_[1][0],
             (ROUTE_LENGTH - veh_sq_[1][-1]),
             # (traci.vehicle.getSpeed(car) - traci.vehicle.getSpeed(veh_sq_[0][0])),
             (den_former - den_later)
             ]
        s_norm = [s[0] / 15,
                  s[1] / 110,
                  s[2] / 110,
                  # s[3] / 20 + 0.5,
                  s[3] / 2000 + 0.5]
        # print('STATE: ', s_norm)
        return s_norm

    @staticmethod
    def _get_density(veh_sq):
        look_range = 100         # default: 200
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
            v_h_for += (1 - math.pow(veh_sq[1][idx] / look_range, 1)) / max(0.1, traci.vehicle.getSpeed(
                car_))  # Weighted Harmonic Sum
            w_v_f += (1 - math.pow(veh_sq[1][idx] / look_range, 1))
            # sum_d_for += (1 - traci.vehicle.getAcceleration(car_) / 3) / veh_sq[1][idx]
            # w_p_f += (1 - traci.vehicle.getAcceleration(car_) / 3)
            if idx == 0:
                sum_d_for += 1 / veh_sq[1][0]
                # w_p_f += 200
                w_p_f += 1
            else:
                sum_d_for += (1 - math.pow(veh_sq[1][idx - 1] / look_range, 1)) / max(0.1, veh_sq[1][idx] - veh_sq[1][idx - 1])
                # w_p_f += (200 - veh_sq[1][idx - 1])
                w_p_f += (1 - math.pow(veh_sq[1][idx - 1] / look_range, 1))
        # v_h_for = max(1 / v_h_for, 0.1)  # Harmonic Mean
        # v_h_for = max(v_h_for / n_former_car, 0.1)  # Arithmetic Mean
        v_h_for = max(w_v_f / v_h_for, 0.1)  # Weighted Harmonic Mean
        for idx, car_ in zip(range(len(later_car_list)), later_car_list):
            # v_h_lat += 1 / max(0.1, traci.vehicle.getSpeed(car_))
            # v_h_lat += max(0.1, traci.vehicle.getSpeed(car_))
            v_h_lat += (1 - math.pow((ROUTE_LENGTH - veh_sq[1][idx - n_later_car]) / look_range, 1)) / max(0.1,
                                                                                           traci.vehicle.getSpeed(car_))
            w_v_l += (1 - math.pow((ROUTE_LENGTH - veh_sq[1][idx - n_later_car]) / look_range, 1))
            # sum_d_lat += (1 - traci.vehicle.getAcceleration(car_) / 3) / (veh_sq[1][idx - n_later_car + 1] - veh_sq[1][idx - n_later_car])
            # w_p_l += (1 - traci.vehicle.getAcceleration(car_) / 3)
            if idx == n_later_car - 1:
                sum_d_lat += 1 / (ROUTE_LENGTH - veh_sq[1][-1])
                # w_p_l += 200
                w_p_l += 1
            else:
                sum_d_lat += (1 - math.pow((ROUTE_LENGTH - veh_sq[1][idx - n_later_car + 1]) / look_range, 1)) / max(0.1, veh_sq[1][
                    idx - n_later_car + 1] - veh_sq[1][idx - n_later_car])
                # w_p_l += (200 - (ROUTE_LENGTH - veh_sq[1][idx - n_later_car + 1]))
                w_p_l += (1 - math.pow((ROUTE_LENGTH - veh_sq[1][idx - n_later_car + 1]) / look_range, 1))
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

        # print('den_former: ', den_former, 'sdf: ', sum_d_for, 'wpf: ', w_p_f, 'vh: ', v_h_for, 'itv: ', veh_sq[1][0])
        # print('den_later: ', den_later, 'sdll: ', sum_d_lat, 'wpl: ', w_p_l, 'vh: ', v_h_lat, 'itv: ', ROUTE_LENGTH - veh_sq[1][-1])
        return min(1000., den_former * 2.5), min(1000., den_later * 2.5), v_h_for, v_h_lat, n_former_car, n_later_car
        # return n_former_car*50, n_later_car * 50

    def _get_reward(self, carID):
        r = 0  # 3. Calculate the reward
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

        if self.s[carID][1] * 110 <= 7:
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
        r -= 30 * abs(self.s[carID][3] * 2000 - 1000)  # for density
        r -= 80 * abs(self.act[carID]) ** 2
        # if carID in traci.simulation.getCollidingVehiclesIDList():
        if self.s[carID][1] * 110 <= 5:
            r -= 10000
        return max(-20000, r) / 20000 - 1

    def _get_reward_test(self, car):
        r = 0
        # rou = abs(self.s[car][3] - 0.5)
        # tendency = np.array(self.past_roue).mean()
        # r = -0.5 * rou - tendency

        v = np.zeros((N_CAR,))
        for car, idx in zip(self.veh_list, range(len(self.veh_list))):
            v[idx] = traci.vehicle.getSpeed(car)
        # r += v.mean() / 5

        h = np.zeros((N_CAR,))
        for car, idx in zip(self.veh_list, range(len(self.veh_list))):
            h[idx] = traci.vehicle.getLeader(car, dist=1000)[1] + 2
            print(traci.vehicle.getLeader(car, dist=1000))

        r = - np.linalg.norm(7 * np.ones((N_CAR,)) - v) - 0.5 * np.sum(12 * np.ones((N_CAR,)) - h)
        # if 5 < self.s[car][1] * 90 <= 7:
        #     r = -0.3
        # elif self.s[car][1] * 90 <= 5:
        #     r = -0.5
        return r

    @staticmethod
    def _get_interval(pos_cur_car, veh_sq):
        veh_sq[1] -= pos_cur_car * np.ones((len(veh_sq[1])))
        for idx in range(len(veh_sq[1])):
            if veh_sq[1][idx] < 0:
                veh_sq[1][idx] += ROUTE_LENGTH
        return list(veh_sq[1])
