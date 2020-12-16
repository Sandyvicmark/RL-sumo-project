import os
import sys
import numpy as np
import math
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle as pk

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci

EPISODE = 30000
START_TURBULENCE = 3000
TURBULENCE_TERM = 500
TEST_TERM = 4000
ACT_SPACE = np.linspace(-3, 2, 50)
N_STATE = 4
N_ACT = ACT_SPACE.size
MEMORY_CAPACITY = 500
TARGET_UPDATE_ITER = 1000
BATCH_SIZE = 256
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(N_STATE, 18)           # build hidden layer
        self.hidden1.weight.data.normal_(0, 0.1)        # initialize params of hidden layer
        self.out = nn.Linear(18, N_ACT)                 # build output layer
        self.out.weight.data.normal_(0, 0.1)            # initialize params of output layer

    def forward(self, x):                               # forward
        x = self.hidden1(x)
        x = F.sigmoid(x)
        act_value = self.out(x)
        return act_value

class DQN:
    def __init__(self):
        use_old_nn = int(raw_input('Use trained network? '))
        if use_old_nn:
            d = int(raw_input('Input network number: '))
            self.eval_net = tc.load('/home/sandymark/RL-sumo/net/evalnet' + str(d) + '.torch')
            self.target_net = tc.load('/home/sandymark/RL-sumo/net/targetnet' + str(d) + '.torch')
        else:
            self.eval_net, self.target_net = Net(), Net()

        self.update_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATE*2 + 2))
        self.optimizer = tc.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_act(self, x):
        x = tc.unsqueeze(tc.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:
            act_value = self.eval_net.forward(x)
            action = tc.max(act_value, 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, N_ACT)
        return action

    def store_transition(self, s, a, r, s_):
        trans = np.hstack((s, [a, r], s_))
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
            b_s = tc.FloatTensor(batch_memory[:, :N_STATE])
            b_a = tc.LongTensor(batch_memory[:, N_STATE:N_STATE + 1].astype(int))
            b_r = tc.FloatTensor(batch_memory[:, N_STATE + 1:N_STATE + 2])
            b_s_ = tc.FloatTensor(batch_memory[:, -N_STATE:])

            q_eval = self.eval_net(b_s).gather(1, b_a)      # Input current STATE to 'eval_net', the net outputs the q_value of each action.
                                                            # Use .gather(1, b_a) to select the q_value of the action that we really chose by e-greedy above,
                                                            # where the 1st arg of gather means search by col(0) or row(1), 2nd arg is an index, to search the value we want to pick up.
            q_next = self.target_net(b_s_).detach()         # .detach() means resist back-propagation from the 'target_net'
                                                            # because we only need to update the 'target_net' in some special steps.
            q_target = b_r + GAMMA * q_next.max(1)[0]       # .max() 'll return a tuple, where 1st element is value and 2nd element is its index.
            loss = self.loss_func(q_eval, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def _get_veh_sequence(carID, route):
    """
        Generate a list, storing the car sequence before dqn-car, in which the closest former car is the first element.
    """
    while True:
        try:
            veh_list = []
            for lane in route:
                veh_list += traci.lane.getLastStepVehicleIDs(lane)
            # veh_list = traci.lane.getLastStepVehicleIDs('edge2_0') + traci.lane.getLastStepVehicleIDs('edge3_0')
            veh_sequence = veh_list[veh_list.index(carID) + 1 : ] + veh_list[: veh_list.index(carID)]
            break
        except ValueError:
            traci.simulationStep()
            continue
    return veh_sequence

def run():
    global EPSILON, dqn_car, route1
    step = 0
    T = 1.5

    dqn = DQN()

    traci.simulationStep()
    edge_list = traci.route.getEdges('route1')
    former_veh = None
    s = {}
    a = {}
    act = {}
    act_old = {}
    # r = {}
    reward = {}
    acc = {}
    v = {}
    headway = {}
    emer_brake_count = {}
    dang_headway_count = {}
    for item in dqn_car:
        s[item] = None
        a[item] = None
        act[item] = None
        act_old[item] = 0.
        # r[item] = None
        reward[item] = []
        acc[item] = []
        v[item] = []
        headway[item] = []
        emer_brake_count[item] = 0
        dang_headway_count[item] = 0

    av_step = 0
    speed_avail = [12, 5, 0, 0, 0, 14, 7, 0, 0, 12, 4, 13, 2, 0, 0, 14, 2, 14, 1]
    while True:                                                         # Waiting for 2 car drive into route
        traci.simulationStep()
        veh_list = traci.vehicle.getIDList()
        print 'VEH_LIST: ', veh_list
        if len(veh_list) != 10:
            continue
        else:
            break

    for car in dqn_car:                                                 # Calculate headway
        print car, '????'
        veh_sq = _get_veh_sequence(car, route1)
        angle_for = math.atan2(traci.vehicle.getPosition(veh_sq[0])[1] - 200, traci.vehicle.getPosition(veh_sq[0])[0])
        angle_lat = math.atan2(traci.vehicle.getPosition(car)[1] - 200, traci.vehicle.getPosition(car)[0])
        delta_angle = angle_for - angle_lat
        if delta_angle < 0:
            delta_angle += 2 * math.pi

        """
        s: current state
        index:  0: speed of dqn-car
                1: headway
                2: speed of leader
                3: difference of dqn-car and leader
        """                                                                 # Generate initial state
        s[car] = [traci.vehicle.getSpeed(car), delta_angle * 100., traci.vehicle.getSpeed(veh_sq[0]),
             traci.vehicle.getSpeed(car) - traci.vehicle.getSpeed(veh_sq[0])]

    while step < EPISODE:                                           # Train start:
        for car in dqn_car:
            a[car] = dqn.choose_act(s[car])                             # 1. Choose an action
            act[car] = ACT_SPACE[a[car]]
            print 'a: ', act
            print 'v: ', s[car][0]
            traci.vehicle.setSpeed(car, max(0, s[car][0]+0.1*act[car]))         # 2. Perform this action

        traci.simulationStep()

        for car in dqn_car:
            veh_sq = _get_veh_sequence(car, route1)
            if car not in traci.simulation.getCollidingVehiclesIDList():# 3. Calculate the reward
                # vf = traci.vehicle.getSpeed(veh_list[0])
                s_star = 5 + 3. + max(0, s[car][0] * T + s[car][0] * (s[car][0] - s[car][2]) / (2 * math.sqrt(2. * 3.)))
                a_star = min(2, max(-3, 2. * (1 - (s[car][0] / 20.) ** 4 - (s_star / s[car][1]) ** 2)))
                r = -100 * abs(act[car] - a_star)

                if ((traci.vehicle.getAcceleration(veh_sq[0]) <= -2 and s[car][1] <= 20) or
                    (s[car][3] >= 5 and s[car][1] <= (s[car][0]**2 - s[car][2]**2) / 2*2.5 + 3)) and act[car] > -2:
                    r -= 1000

                r -= min(400, abs(s[car][3]) ** 4)
                r -= min(50, 10 * abs(act[car] - act_old[car]))

            else:
                r = -10000
            # r[car] = _r

            print 'INTERVAL: ', car, s[car][1]
            reward[car].append(r)
            acc[car].append(act[car])
            v[car].append(s[car][0])
            headway[car].append(s[car][1])
            print 'REWARD: ', car, r

                                                                        # 4. Get the next state from sumo
            angle_for = math.atan2(traci.vehicle.getPosition(veh_sq[0])[1] - 200, traci.vehicle.getPosition(veh_sq[0])[0])
            angle_lat = math.atan2(traci.vehicle.getPosition(car)[1] - 200, traci.vehicle.getPosition(car)[0])
            delta_angle = angle_for - angle_lat
            if delta_angle < 0:
                delta_angle += 2 * math.pi
            s_ = [traci.vehicle.getSpeed(car), delta_angle * 100., traci.vehicle.getSpeed(veh_sq[0]),
                  traci.vehicle.getSpeed(car) - traci.vehicle.getSpeed(veh_sq[0])]

            dqn.store_transition(s[car], a[car], r, s_)                 # 5. Store the data

            act_old[car] = act[car]
            s[car] = s_

            if step >= EPISODE - TEST_TERM:                             # Bad behaviour count
                EPSILON = 1.
                if act[car] < -2:
                    emer_brake_count[car] += 1
                if s[car][1] < 7:
                    dang_headway_count[car] += 1

        if dqn.memory_counter > MEMORY_CAPACITY:                        # 6. Train the eval-net
            dqn.learn()

        step += 1
        av_step += 1

        """ Manually change the leader's behaviour to train the dqn-car """
        if step % TURBULENCE_TERM == 0 and step >= START_TURBULENCE:
            traci.vehicle.setSpeed('0', speed_avail[(step / TURBULENCE_TERM) % len(speed_avail)])

        # if step == EPISODE - 2000:
        #     traci.vehicle.setSpeed('0', 12)
        # elif step == EPISODE - 1500:
        #     traci.vehicle.setSpeed('0', 2)
        # elif step == EPISODE - 1000:
        #     traci.vehicle.setSpeed('0', 13)
        # elif step == EPISODE - 500:
        #     traci.vehicle.setSpeed('0', 1)

    return reward, acc, v, headway, emer_brake_count, dang_headway_count, av_step, dqn.eval_net, dqn.target_net


if __name__ == '__main__':
    sumoBinary = 'sumo-gui'
    sumoCmd = [sumoBinary, '-c', '/home/sandymark/RL-sumo/net.sumocfg']

    traci.start(sumoCmd)
    dqn_car = ['999', '998', '997', '996']
    # route1 = ['80_0', '81_0', '82_0', '83_0', '84_0', '85_0']
    route1 = ['edge2_0', 'edge3_0']
    re, a, vel, hdway, e_b, d_h, t, eval_net, target_net = run()
    for i in range(len(dqn_car)):
        plt.figure(i)
        plt.plot(np.linspace(0, t, t), re[dqn_car[i]], 'b')
        plt.plot(np.linspace(0, t, t), a[dqn_car[i]], 'r')
        plt.plot(np.linspace(0, t, t), vel[dqn_car[i]], 'g')
        plt.plot(np.linspace(0, t, t), hdway[dqn_car[i]], 'y')
        plt.show()
        a_part = np.array(a[dqn_car[i]][500:])
        a_std = a_part.std()
        print 'a_STD: ', dqn_car[i], a_std
        print 'Emergency brake performed: ', e_b[dqn_car[i]]
        print 'Dangerous headway: ', d_h[dqn_car[i]]
    save = int(raw_input('Save this NN? (1/0)\n'))
    if save:
        num = int(raw_input('Input the number: '))
        tc.save(eval_net, '/home/sandymark/RL-sumo/net/evalnet' + str(num) +'.torch')
        tc.save(target_net, '/home/sandymark/RL-sumo/net/targetnet' + str(num) +'.torch')

