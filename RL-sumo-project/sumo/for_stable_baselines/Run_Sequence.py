# import stable_baselines as sb
import stable_baselines3 as sb3
import matplotlib.pyplot as plt
import torch
from for_stable_baselines.Env_singlecar_test import CustomEnv


def pre_train(step, env, policy_args=None, save=False, save_dir=None):
    if policy_args is None:
        model = sb3.PPO('MlpPolicy', env, 0.003, vf_coef=0.5, batch_size=64, gamma=0.9, verbose=1)
        # model = sb.DQN('MlpPolicy', env, gamma=0.9, learning_rate=0.001, batch_size=64, verbose=1)
    else:
        model = sb3.PPO('MlpPolicy', env, 0.003, vf_coef=0.5, batch_size=64, gamma=0.9, verbose=1, policy_kwargs=policy_args)
        # model = sb.DQN('MlpPolicy', env, gamma=0.9, learning_rate=0.001, batch_size=64, verbose=1, policy_kwargs=policy_args)
    model.learn(total_timesteps=step)
    if save:
        model.save('/Users/sandymark/StableBaselines_TrainResult/' + save_dir)


def re_train(step, env, load_dir=None, save=True, save_dir=None):
    assert load_dir is not None, TypeError
    model = sb3.PPO.load('/Users/sandymark/StableBaselines_TrainResult/' + load_dir, env)
    model.learn(total_timesteps=step)
    if save:
        model.save('/Users/sandymark/StableBaselines_TrainResult/' + save_dir)


def test(step, env, load_name=None):
    assert load_name is not None, TypeError
    s = env.reset(is_test=True)
    # model = sb3.PPO.load('/Users/sandymark/StableBaselines_TrainResult/ppo_ring_singlecar_stopandgowave_reward3/' + load_name, env)
    # for n in range(step):
    #     a = model.predict(s)
    #     s_, _, _, _ = env.step(a)
    #     s = s_
    for n in range(step):
        env.step([0])


if __name__ == '__main__':
    plt.ion()
    for i in range(2):
        env = CustomEnv('')
        policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=[{'pi': [96, 96], 'vf': [64, 64]}])
        # pre_train(300000, env, policy_kwargs, False, 'ppo_ring_2car_stopandgowave_reward3/ppo_ring' + str(i) + '_0')
        test(10000, env, 'ppo_ring0_0')

    # for i in range(29):
    #     re_train(10000, env, 'ppo_ring1_' + str(i), True, 'ppo_ring1_' + str(i + 1))
    # test(10000, env, 'ppo_ring_19')

    # model = sb3.PPO('MlpPolicy',
    #                 env,
    #                 learning_rate=0.0003,
    #                 n_epochs=10,
    #                 n_steps=1024,
    #                 batch_size=64,
    #                 gamma=0.9,
    #                 policy_kwargs=policy_kwargs,
    #                 verbose=1)
    # model = sb3.PPO('MlpPolicy', env, 0.003, vf_coef=0.1, batch_size=64, gamma=0.9, verbose=1)
    # model = sb3.PPO.load('/Users/sandymark/StableBaselines_TrainResult/ppo_sqs', env)
    # model = sb.PPO2('MlpPolicy', policy_kwargs=policy_kwargs, env=env, gamma=0.9, learning_rate=0.05, verbose=2, ent_coef=0.02)
    # model = sb.DDPG('MlpPolicy', env)
    # model = sb.TRPO('MlpPolicy', env, vf_stepsize=0.01, policy_kwargs=policy_kwargs)
    # model = sb.A2C('MlpPolicy', env, policy_kwargs=policy_kwargs)
    # model.learn(total_timesteps=50000)
    # model.save('/Users/sandymark/StableBaselines_TrainResult/ppo_sqs_1')
    # observer = Observer()
    # observer.plot_var(env.)
    # tune.run('PPO', config={'env': CustomEnv, 'num_workers': 4,})
    # model = sb3.PPO.load('/Users/sandymark/StableBaselines_TrainResult/ppo_sqs')
    # s = env.reset()
    # while True:
    #     a = model.predict(s)
    #     s_, _, _, _ = env.step(a)
    #     s = s_

    plt.show()
    plt.ioff()
