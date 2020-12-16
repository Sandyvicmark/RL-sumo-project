"""
Function Set Instrument:
    This file contains some common functions, which can retrieve some info from SUMO or
    execute a set of control commands to SUMO.
    All functions are self-made functions, which may has a certain of instability to
    the program where it used.
"""
import os
import sys
import numpy as np

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci


def wait_all_vehicles(n_car):
    while True:
        traci.simulationStep()
        veh_list = list(traci.vehicle.getIDList())
        traci.vehicle.setSpeed('0', 0.2)
        if len(veh_list) != n_car:
            continue
        else:
            traci.vehicle.setSpeed('0', -1)
            break
    print('SUMOSDK [I]: All vehicles loaded!')
    veh_list = sorted(veh_list, key=lambda x: int(x), reverse=False)
    return veh_list


def apply_acceleration(car: str, action: list or object, step_length: float, do_step: False):
    try:
        a = action[0]
    except:
        a = action
    v0 = traci.vehicle.getSpeed(car)
    traci.vehicle.setSpeed(car, v0 + a * step_length)
    if do_step:
        traci.simulationStep()


def make_turbulence(tur_car, step, start_tur, tur_term, tur_span, tur_scale=0.7,
                    random_car=False, random_scale=0., veh_list=None):
    if step >= start_tur and step % tur_term == 0:
        if random_car:
            assert isinstance(veh_list, list), TypeError
            tur_car = np.random.choice(np.array(veh_list), 1)[0]
        else:
            assert isinstance(tur_car, str), TypeError
        tur_v = np.clip(0, 1, 1 - random_scale * np.random.random(1)[0] - tur_scale) \
                * traci.vehicle.getSpeed(tur_car)
        traci.vehicle.setSpeed(tur_car, tur_v)
    if step >= start_tur and step % tur_term == tur_span:
        traci.vehicle.setSpeed(tur_car, -1)


def get_veh_sequence(route, car, n_car):
    """ Generate a list, storing the car sequence before dqn-car,
        in which the closest former car is the first element. """
    while True:
        try:
            while True:
                veh_list = [[], []]
                for lane in route:
                    veh_list[0] += traci.lane.getLastStepVehicleIDs(lane)
                if len(veh_list[0]) != n_car:
                    traci.simulationStep()
                    continue
                else:
                    veh_list[0] = veh_list[0][veh_list[0].index(car) + 1:] + veh_list[0][: veh_list[0].index(car)]
                    for item in veh_list[0]:
                        veh_list[1].append(get_interval(route, car, item))
                    break
            break
        except ValueError:
            traci.simulationStep()
            print('ValueError')
            continue
    return veh_list


def get_interval(route, car, leader_car):
    car_lane_index = route.index(traci.vehicle.getLaneID(car))
    leader_car_lane_index = route.index(traci.vehicle.getLaneID(leader_car))
    if leader_car_lane_index < car_lane_index:
        leader_car_lane_index += len(route)
    itv = 0.
    for n_lane in range(car_lane_index, leader_car_lane_index):
        itv += traci.lane.getLength(route[n_lane % len(route)])
    itv -= traci.vehicle.getLanePosition(car)
    itv += traci.vehicle.getLanePosition(leader_car)
    if abs(itv) <= 1e-6:
        itv = 1e-6
    if itv < 0:
        for lane in route:
            itv += traci.lane.getLength(lane)
    return itv
