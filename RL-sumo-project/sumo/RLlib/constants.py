# discount rate
GAMMA = 0.9

# for clipping ratios
EPSILON = 0.2

# lambda constant for GAE
TAU = 0.95

# number of episodes to train
N_EPISODES = 300

# number of frames to store in memory
BATCH_SIZE = 5000

# number of hidden units in models of actor & critic
N_HIDDEN = 64

# learning rate for adam optimizer
A_LEARNING_RATE = 0.001
C_LEARNING_RATE = 0.01

# interval of steps after which statistics should be printed
LOG_STEPS = 1

# interval of steps after which models should be saved
SAVE_STEPS = 20

# path to save actor model
ACTOR_SAVE_PATH = "saved_models/actor_ppo.pth"

# path to sace critic model
CRITIC_SAVE_PATH = "saved_models/critic_ppo.pth"
