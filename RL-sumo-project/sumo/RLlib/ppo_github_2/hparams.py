class HyperParams:
    gamma = 0.9
    lamda = 0.95
    hidden = 64
    critic_lr = 0.0005
    actor_lr = 0.0005
    batch_size = 64
    l2_rate = 0.001
    max_kl = 0.01
    clip_param = 0.2

    save_policy = True
    train_mode = [True, False]
    test_policy_dir = '/home/sandymark/SnG_wave_TrainResult/policy/2020120800-03-55'
