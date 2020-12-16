import traci
import math
import time
import numpy as np
import pandas as pd
import gym
from Tools import SumoSDK
from Tools.Statistics import Observer
# from ray import tune
# import ray
# from ray.rllib.agents.ppo import PPOTrainer

N_CAR = 25
RL_CAR = ['0', '12']
N_RL_CAR = len(RL_CAR)
ROUTE = ['edge290_0', 'edge390_0']
ROUTE_LENGTH = 283.


class CustomEnv(gym.Env):
    def __init__(self, _):
        super(CustomEnv, self).__init__()
        self.rl_car = RL_CAR
        self.n_rl_car = N_RL_CAR
        self.action_space = gym.spaces.Box(low=-np.ones((1,)),
                                           high=np.ones((1,)),
                                           shape=(1,),
                                           dtype=np.float)
        # self.action_space = gym.spaces.Discrete(40)
        self.observation_space = gym.spaces.Box(low=np.zeros((4,)),
                                                high=np.ones((4,)),
                                                shape=(4,),
                                                dtype=np.float)
        self.done = list(np.zeros(self.n_rl_car, dtype=np.bool))
        self.is_test = False
        self.reset_pos_x = [45.0, 44.276831486938335, 42.13056917878817, 38.630195712083975, 33.88821597016249,
                            28.05704108364301, 21.324089811284942, 13.905764746872634, 6.040496961794498,
                            -2.018917365773159, -10.013442028034145, -17.68612642442656, -24.79036416534461,
                            -31.097819204408903, -36.40576474687263, -40.54359905560886, -43.378328731313395,
                            -44.818843229785756, -44.818843229785756, -43.3783287313134, -40.54359905560886,
                            -36.405764746872634, -31.097819204408914, -24.79036416534462, -17.68612642442657,
                            -10.013442028034156, -2.01891736577318, 6.040496961794487, 13.905764746872626,
                            21.324089811284935, 28.057041083643, 33.88821597016249, 38.63019571208397,
                            42.13056917878817, 44.276831486938335]

        self.reset_pos_y = [200.0, 208.03506026593865, 215.81186708366042, 223.08046748326578, 229.60724266728707,
                            235.18241671106134, 239.6267989335532, 242.7975432332819, 244.59273927955707,
                            244.95468799435918, 243.87175604818208, 241.37874976481527, 237.55579641745862,
                            232.52576887223262, 226.4503363531613, 219.52476826029013, 211.9716580505004,
                            204.0337689006545, 195.96623109934552, 188.02834194949963, 180.4752317397099,
                            173.54966364683872, 167.47423112776738, 162.44420358254138, 158.62125023518473,
                            156.12824395181795, 155.04531200564085, 155.40726072044293, 157.2024567667181,
                            160.37320106644677, 164.81758328893866, 170.39275733271293, 176.91953251673422,
                            184.18813291633955, 191.96493973406135]
        self.turb_car = str(np.random.randint(1, N_CAR - 1, 1)[0])
        self.route = ['edge290_0', 'edge390_0']
        self.record_r = []
        self.record_vmean = []
        self.record_vstd = []
        self.observer = Observer()
        self.n_step = 0
        sumo_binary = 'sumo'
        sumo_cmd = [sumo_binary, '-c', '/home/sandymark/RL-sumo/net.sumocfg', '--collision.action', 'warn']
        traci.start(sumo_cmd)

        self.veh_list = SumoSDK.wait_all_vehicles(N_CAR)

    def step(self, action):
        # print(action)
        # SumoSDK.apply_acceleration('0', action, 0.1, do_step=True)
        for car_, idx in zip(RL_CAR, range(len(RL_CAR))):
            SumoSDK.apply_acceleration(car_, action[idx] * 2.5 - 0.5, 0.1, do_step=False)
        traci.simulationStep()

        s_ = self._get_state()
        r = self._get_reward(s_, action)
        # v_ = traci.vehicle.getSpeed('0')
        self.record_r.append(np.mean(r))

        for idx in range(self.n_rl_car):
            if self.rl_car[idx] in traci.simulation.getCollidingVehiclesIDList():
                self.done[idx] = True

        if True in self.done:
            _ = self.reset(reset_stats=False)

        # if self.n_step > 50000 and self.n_step % 500:
        #     # if self.last_turb_car is not None:
        #     #     traci.vehicle.setSpeed(self.last_turb_car, -1)
        #
        #     # self.turb_car = str(np.random.randint(1, 24, 1)[0])
        #     self.turb_car = str(np.random.randint(1, 24, 1)[0]) \
        #         if str(np.random.randint(1, 24, 1)[0]) != '12' else str(np.random.randint(13, 24, 1)[0])
        #     traci.vehicle.setSpeed(self.turb_car, 0.4)
        # if self.n_step > 50000 and self.n_step % 81:
        #     traci.vehicle.setSpeed(self.turb_car, -1)

        # self.observer.plot_var_dyn([r, action[0], s_[3] - 0.5], self.n_step, 1000, [0, 0, 0], 0, ['b', 'r', 'y'])

        self.n_step += 1

        if self.n_step % 5000 == 0 and self.n_step > 0:
            print('#' * 30)
            print('Episode mean reward: ', np.array(self.record_r)[self.n_step - 1000: self.n_step].mean())
            print('STEP pass: ', self.n_step)
            print('#' * 30)

        # if not self.is_test and self.n_step % 300000 == 0:
        #     time_str = time.strftime('%Y%m%d%H-%M-%S', time.localtime(time.time()))
        #     df = pd.DataFrame(np.array(self.record_r))
        #     df.to_csv('/home/sandymark/SnG_wave_TrainResult/' + time_str + '.csv', index=False)

        if self.is_test and self.n_step % 10000 == 0:
            time_str = time.strftime('%Y%m%d%H-%M-%S', time.localtime(time.time()))
            df_test = pd.DataFrame(np.array(self.record_vmean))
            df_test.to_csv('/home/sandymark/SnG_wave_TestResult/vmean_' + time_str + '.csv', index=False)
            df_test = pd.DataFrame(np.array(self.record_vstd))
            df_test.to_csv('/home/sandymark/SnG_wave_TestResult/vstd_' + time_str + '.csv', index=False)
        return s_, r, self.done, (self.record_vmean, self.record_vstd)

    def reset(self, is_test=False, return_state=True, reset_stats=False):
        s = []
        if is_test:
            self.is_test = True
        self.done = list(np.zeros(self.n_rl_car, dtype=np.bool))
        x = self.reset_pos_x
        y = self.reset_pos_y
        reset_idx = list(np.random.choice(len(x), (N_CAR,), replace=False))
        # veh_list = sorted(self.veh_list, key=lambda i: int(i))

        for idx, item in zip(range(len(reset_idx)), reset_idx):
            if item < (len(x) // 2):
                edge = 'edge290'
            else:
                edge = 'edge390'
            traci.vehicle.moveToXY(self.veh_list[idx], edge, 0, x[item], y[item])
            traci.vehicle.setSpeed(self.veh_list[idx], 0)

        traci.simulationStep()

        for car_ in self.veh_list:
            traci.vehicle.setSpeed(car_, -1)

        if reset_stats:
            self.record_vstd = []
            self.record_vmean = []

        if return_state:
            veh_sq = self._get_veh_sequence()  # Generate initial state
            # for car_ in RL_CAR:
            #     car_idx = veh_sq[0].index(car_)
            #     veh_sq_ = [veh_sq[0][car_idx + 1:] + veh_sq[0][: car_idx],
            #                veh_sq[1][car_idx + 1:] + veh_sq[1][: car_idx]]
            #     veh_sq_[1] = self._get_interval(veh_sq[1][car_idx], veh_sq_)
            s = self._get_state()
            # get centralized state
            # s = []
            # for _ in RL_CAR:
            #     s += self.s[_]
            return s

    def render(self, mode='human'):
        pass

    def _get_state(self):
        s = []
        # -
        # v = []
        # for car_ in self.veh_list:
        #     v.append(traci.vehicle.getSpeed(car_))
        # -
        veh_sq = self._get_veh_sequence()
        for car_ in RL_CAR:
            v = traci.vehicle.getSpeed(car_)
            car_idx = veh_sq[0].index(car_)
            veh_sq_ = [veh_sq[0][car_idx + 1:] + veh_sq[0][: car_idx],
                       veh_sq[1][car_idx + 1:] + veh_sq[1][: car_idx]]
            veh_sq_[1] = self._get_interval(veh_sq[1][car_idx], veh_sq_)
            den_f, den_l, _, _, _, _ = self._get_density(veh_sq_)
            # print(den_f, den_l)
            s.append([v / 15,
                      veh_sq_[1][0] / 173.,
                      (v - traci.vehicle.getSpeed(veh_sq_[0][0]) + 10) / 25.,
                      (den_f - den_l + 1000) / 2000.,
                      ])
        # -
        # pos_dict = {}
        # for car, pos in zip(veh_sq[0], veh_sq[1]):
        #     pos_dict[car] = pos
        # pos_dict_sorted = dict(sorted(pos_dict.items(), key=lambda x:x[1], reverse=False))
        # for car in pos_dict_sorted:
        #     s.append(v[int(car)] / 15.)
        #     s.append(pos_dict_sorted[car] / 283.)
        # -
        return s

    def _get_reward(self, state, action):
        v = []
        veh_list_copy = self.veh_list[:]
        veh_list_copy.remove(self.turb_car)
        for car_ in veh_list_copy:
            v.append(traci.vehicle.getSpeed(car_))
        v_mean = np.array(v).mean()
        v_std = np.array(v).std()
        self.record_vmean.append(v_mean)
        self.record_vstd.append(v_std)

        r = []
        for idx in range(N_RL_CAR):
            # method 1: use rho_e and dangerous headway punishment
            # r = - abs(5 * (state[3] * 2 - 1) + 2 * min(1., max(0, 7 - max(0, state[1] * 173)) / 6.))

            # method 2: use v_mean & std of platoon and dangerous headway punishment
            # r = 1.5 * v_mean / 10. - 0.2 * v_std / 2. \
            #     - 1. * (min(1., max(0., 7 - max(0., state[1] * 173)) / 6.)
            #              )  # + min(1., max(0., 7 - max(0., state[5] * 173)) / 6.)

            # method 3: use v_mean & rho_e and dangerous headway punishment
            r_ = 1. - 0 * v_mean / 3. \
                 - (3 * abs(state[idx][3] * 2 - 1)
                    + 0.3 * min(1., max(0, 7 - max(0, state[idx][1] * 173)) / 6.))

            # method 4: use norm of vector v to encourage system-level speed
            # r = (np.linalg.norm(3. * np.ones_like(v)) - np.linalg.norm(3. * np.ones_like(v) - np.array(v))) \
            #     / np.linalg.norm(3. * np.ones_like(v))
            if self.rl_car[idx] in traci.simulation.getCollidingVehiclesIDList():
                r_ = -50
            r.append(100 * r_)

        return r

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

    @staticmethod
    def _get_interval(pos_cur_car, veh_sq):
        veh_sq[1] -= pos_cur_car * np.ones((len(veh_sq[1])))
        for idx in range(len(veh_sq[1])):
            if veh_sq[1][idx] < 0:
                veh_sq[1][idx] += ROUTE_LENGTH
        return list(veh_sq[1])

    @staticmethod
    def _get_density(veh_sq):
        look_range = 100  # default: 200
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
            v_h_for += (1 - math.pow(veh_sq[1][idx] / look_range, 1)) \
                       / max(0.5, traci.vehicle.getSpeed(car_))  # Weighted Harmonic Sum
            w_v_f += (1 - math.pow(veh_sq[1][idx] / look_range, 1))
            if idx == 0:
                sum_d_for += 1 / veh_sq[1][0]
                w_p_f += 1
            else:
                sum_d_for += (1 - math.pow(veh_sq[1][idx - 1] / look_range, 1)) \
                             / max(0.1, veh_sq[1][idx] - veh_sq[1][idx - 1])
                w_p_f += (1 - math.pow(veh_sq[1][idx - 1] / look_range, 1))
        v_h_for = max(w_v_f / v_h_for, 0.1)  # Weighted Harmonic Mean
        for idx, car_ in zip(range(len(later_car_list)), later_car_list):
            v_h_lat += (1 - math.pow((ROUTE_LENGTH - veh_sq[1][idx - n_later_car]) / look_range, 1)) \
                       / max(0.5, traci.vehicle.getSpeed(car_))
            w_v_l += (1 - math.pow((ROUTE_LENGTH - veh_sq[1][idx - n_later_car]) / look_range, 1))
            if idx == n_later_car - 1:
                sum_d_lat += 1 / (ROUTE_LENGTH - veh_sq[1][-1])
                w_p_l += 1
            else:
                sum_d_lat += (1 - math.pow((ROUTE_LENGTH - veh_sq[1][idx - n_later_car + 1]) / look_range, 1)) \
                             / max(0.1, veh_sq[1][idx - n_later_car + 1] - veh_sq[1][idx - n_later_car])
                w_p_l += (1 - math.pow((ROUTE_LENGTH - veh_sq[1][idx - n_later_car + 1]) / look_range, 1))
        v_h_lat = max(w_v_l / v_h_lat, 0.1)
        if n_former_car:
            # den_former = n_former_car + 5 * (1 - math.pow(veh_sq[1][0] / 600, 1)) / v_h_for + 1000 / max(0.1, veh_sq[1][0])
            den_former = 300 * sum_d_for / w_p_f + 50 / v_h_for + 800 / max(0.1, veh_sq[1][0])
        else:
            den_former = 0
        if n_later_car:
            # den_later = n_later_car + 5 * (1 - math.pow((629 - veh_sq[1][-1]) / 600, 1)) / v_h_lat + 1000 / max(0.1, 629 - veh_sq[1][-1])
            den_later = 300 * sum_d_lat / w_p_l + 50 / v_h_lat + 800 / max(0.1, (ROUTE_LENGTH - veh_sq[1][-1]))
        else:
            den_later = 0

        # print('den_former: ', den_former, 'sdf: ', sum_d_for, 'wpf: ', w_p_f, 'vh: ', v_h_for, 'itv: ', veh_sq[1][0])
        # print('den_later: ', den_later, 'sdll: ', sum_d_lat, 'wpl: ', w_p_l, 'vh: ', v_h_lat, 'itv: ', ROUTE_LENGTH - veh_sq[1][-1])
        return max(0., min(1000., den_former * 2.5)), max(0., min(1000., den_later * 2.5)), \
            v_h_for, v_h_lat, n_former_car, n_later_car

# if __name__ == '__main__':
    # env = gym.make('Pendulum-v0')
    # ray.init()
    # trainer = ppo.PPOTrainer(env='Pendulum-v0', config={'env_config': {'render': True}, 'verbose': 1})
    # while True:
    #     trainer.train()
    # tune.run(
    #     PPOTrainer,
    #     config={
    #         'env': 'Pendulum-v0',
    #         'lambda': 0.95,
    #         'clip_param': 0.2,
    #         'train_batch_size': 2000,
    #         'sgd_minibatch_size': 256,
    #         'lr': 1e-3,
    #         'vf_loss_coeff': 0.1,
    #         'num_workers': 8,
    #
    #         'log_level': 'INFO'
    #     }
    # )
