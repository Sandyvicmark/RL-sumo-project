import os
import gym
import torch
import argparse
import numpy as np
import torch.optim as optim
from RLlib.ppo_github_2.model import Actor, Critic
from RLlib.ppo_github_2.utils.utils import get_action, save_checkpoint
from collections import deque
# from utils.running_state import ZFilter
from RLlib.ppo_github_2.hparams import HyperParams as hp
# from tensorboardX import SummaryWriter
from RLlib.ppo_github_2.ppo_gae import train_model
from for_stable_baselines.Env_singlecar_test import CustomEnv


if __name__ == "__main__":
    # you can choose other environments.
    # possible environments: Ant-v2, HalfCheetah-v2, Hopper-v2, Humanoid-v2,
    # HumanoidStandup-v2, InvertedPendulum-v2, Reacher-v2, Swimmer-v2,
    # Walker2d-v2
    env = CustomEnv('')
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    print('state size:', num_inputs)
    print('action size:', num_actions)

    # writer = SummaryWriter(args.logdir)

    actor = Actor(num_inputs, num_actions)
    critic = Critic(num_inputs)

    # running_state = ZFilter((num_inputs,), clip=5)

    # if args.load_model is not None:
    #     saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
    #     ckpt = torch.load(saved_ckpt_path)
    #
    #     actor.load_state_dict(ckpt['actor'])
    #     critic.load_state_dict(ckpt['critic'])
    #
    #     running_state.rs.n = ckpt['z_filter_n']
    #     running_state.rs.mean = ckpt['z_filter_m']
    #     running_state.rs.sum_square = ckpt['z_filter_s']
    #
    #     print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr, weight_decay=hp.l2_rate)

    episodes = 0
    score_avg_record = []
    for iter in range(100):
        actor.eval(), critic.eval()
        memory = deque()
        steps = 0
        scores = []
        state = env.reset()
        while steps < 2048:
            episodes += 1
            # This LINE has been moved to 2 lines above!
            # state = running_state(state)
            score = 0
            for _ in range(2048):
                steps += 1
                mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, std)[0]
                next_state, reward, done, _ = env.step(action)
                # next_state = running_state(next_state)
                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append([state, action, reward, mask])

                score += reward
                state = next_state

                if done:
                    break
            scores.append(score)
        score_avg = np.mean(scores)
        score_avg_record.append(score_avg)
        print('{} episode score is {:.2f}'.format(episodes, score_avg))
        # writer.add_scalar('log/score', float(score_avg), iter)

        actor.train(), critic.train()
        train_model(actor, critic, memory, actor_optim, critic_optim)

        # if iter % 100:
        #     score_avg = int(score_avg)
        #
        #     model_path = os.path.join(os.getcwd(),'save_model')
        #     if not os.path.isdir(model_path):
        #         os.makedirs(model_path)
        #
        #     ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')
        #
        #     save_checkpoint({
        #         'actor': actor.state_dict(),
        #         'critic': critic.state_dict(),
        #         'z_filter_n':running_state.rs.n,
        #         'z_filter_m': running_state.rs.mean,
        #         'z_filter_s': running_state.rs.sum_square,
        #         'args': args,
        #         'score': score_avg
        #     }, filename=ckpt_path)
    print(score_avg_record)
