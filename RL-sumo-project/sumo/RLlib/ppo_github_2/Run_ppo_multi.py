import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from RLlib.ppo_github_2.model import Actor, Critic
from RLlib.ppo_github_2.utils.utils import get_action
from collections import deque
from RLlib.ppo_github_2.hparams import HyperParams as Hp
from RLlib.ppo_github_2.ppo_gae_multi import train_model
from for_stable_baselines.Env_multicar_test import CustomEnv
import time


def train():
    actor = Actor(num_inputs, num_actions)
    critic = Critic(num_inputs)

    actor_optim = optim.Adam(actor.parameters(), lr=Hp.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=Hp.critic_lr, weight_decay=Hp.l2_rate)

    episodes = 0
    score_avg_record = []
    episode_v_mean = []
    episode_v_std = []

    for itr in range(200):
        actor.eval(), critic.eval()
        memory = deque()
        for _ in range(num_agents):
            memory.append([])
        steps = 0
        scores = []
        state = env.reset(reset_stats=True)
        while steps < 2048:
            episodes += 1
            score = 0
            for _ in range(2048):
                steps += 1
                action = []
                for idx in range(num_agents):
                    mu, std, _ = actor(torch.tensor(state[idx]).unsqueeze(0))
                    single_action = get_action(mu, std)[0]
                    action.append(single_action)

                next_state, reward, done, info = env.step(action)

                for idx in range(num_agents):
                    if done[idx]:
                        mask = 0
                    else:
                        mask = 1
                    memory[idx].append([state[idx], action[idx], reward[idx], mask])

                score += np.mean(reward)
                state = next_state

                if True in done:
                    break
            scores.append(score)
        episode_v_mean.append(np.mean(info[0]))
        episode_v_std.append(np.mean(info[1]))
        score_avg = np.mean(scores)
        score_avg_record.append(score_avg)
        print('{} episode score is {:.2f}'.format(episodes, score_avg))
        print('{} episode evm is {:.2f}'.format(episodes, np.mean(info[0])))
        print('{} episode evs is {:.2f}'.format(episodes, np.mean(info[1])))
        print('{} episode ms ratio is {:.2f}'.format(episodes, np.mean(info[0]) / np.mean(info[1])))
        print('-'*30)

        actor.train(), critic.train()
        train_model(actor, critic, memory, actor_optim, critic_optim)

    print(score_avg_record)
    if Hp.save_policy:
        reward_pd = pd.DataFrame(np.array(score_avg_record))
        v_mean = pd.DataFrame(episode_v_mean)
        v_std = pd.DataFrame(episode_v_std)
        ms_ratio = pd.DataFrame(list(np.divide(episode_v_mean, episode_v_std)))
        time_str = time.strftime('%Y%m%d%H-%M-%S', time.localtime(time.time()))
        with open('/home/sandymark/SnG_wave_TrainResult/policy/' + time_str + 'A.plc', 'wb') as FILE:
            torch.save(actor, FILE)
        with open('/home/sandymark/SnG_wave_TrainResult/policy/' + time_str + 'C.plc', 'wb') as FILE:
            torch.save(critic, FILE)
        reward_pd.to_csv('/home/sandymark/SnG_wave_TrainResult/' + time_str + 'reward.csv', index=False)
        v_mean.to_csv('/home/sandymark/SnG_wave_TrainResult/' + time_str + 'vmean.csv', index=False)
        v_std.to_csv('/home/sandymark/SnG_wave_TrainResult/' + time_str + 'vstd.csv', index=False)
        ms_ratio.to_csv('/home/sandymark/SnG_wave_TrainResult/' + time_str + 'ratio.csv', index=False)


def test(plc_dir, test_step):
    with open(plc_dir + 'A.plc', 'rb') as FILE:
        actor = torch.load(FILE)

    state = env.reset(reset_stats=True)
    for step in range(test_step):
        action = []
        for idx in range(num_agents):
            mu, std, _ = actor(torch.tensor(state[idx]).unsqueeze(0))
            single_action = get_action(mu, std)[0]
            action.append(single_action)

        next_state, _, _, info = env.step(action)
        state = next_state
    v_mean = info[0]
    v_std = info[1]
    ms_ratio = list(np.divide(v_mean, v_std))

    v_mean_pd = pd.DataFrame(v_mean)
    v_std_pd = pd.DataFrame(v_std)
    ms_ratio_pd = pd.DataFrame(ms_ratio)
    time_str = time.strftime('%Y%m%d%H-%M-%S', time.localtime(time.time()))
    v_mean_pd.to_csv('/home/sandymark/SnG_wave_TestResult/' + time_str + 'vmean.csv', index=False)
    v_std_pd.to_csv('/home/sandymark/SnG_wave_TestResult/' + time_str + 'vstd.csv', index=False)
    ms_ratio_pd.to_csv('/home/sandymark/SnG_wave_TestResult/' + time_str + 'ratio.csv', index=False)


if __name__ == "__main__":
    env = CustomEnv('')
    env.seed(500)
    torch.manual_seed(500)
    train_mode = Hp.train_mode[0]

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    num_agents = env.n_rl_car

    print('state size: ', num_inputs)
    print('action size: ', num_actions)
    print('agents: ', num_agents)

    for i in range(4):
        if train_mode:
            train()
        else:
            policy_dir = Hp.test_policy_dir
            test(policy_dir, 10000)
