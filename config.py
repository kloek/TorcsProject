
#### SETTINGS FOR RUN_DDPG #####
# run related parameters!
is_training = True  # False just gives testrunns
test_frequency = 20 # how often to test /episodes
epsilon_start = 1  # TODO sys arg or config file
episode_count = 2000  # TODO sys arg or config file
max_steps = 2000  # TODO sys arg or config file
EXPLORE = 400000.0


# Loggin related parameters
RUN_FOLDER = "runs/" # use if ~/ outside gymtorcs folder
log_size = 100 # number of episodes per log
log_in_file = True
log_memory = False

# Gym_torcs
vision = False
throttle = True
gear_change = False #False = drive only on first gear, limited to 80 km/h
safety_critic = False  # false = normal ddpg, True = double critic


# 1. original sensors!!!
state_dim = 29
## to be able to set sensors from config create state has to be replaced with create state2, but that one is abit slower!!!!
sensor_list = ['angle', 'track', 'trackPos', 'speedX', 'speedY', 'speedZ', 'wheelSpinVel', 'rpm']

# 2. realistic sensors!! (vithout vision)
#state_dim = 89

# 3. combo! for driving without vision, but close to realistic!
#state_dim = 90

# 4. combo! same as 3 but without focus sensor!!!
#state_dim = 85


"""names = ['angle','curLapTime','damage','distFromStart','distRaced','focus','fuel','gear','lastLapTime','opponents','racePos',
     'rpm','speedX','speedY','speedZ','track','trackPos','wheelSpinVel','z']"""

action_dim = 3


#### SETTINGS FOR DDPG_AGENT #####

# Hyper Parameters:
REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 100
BATCH_SIZE = 32  # size of minibatches to train with
GAMMA = 0.99  # γ discount factor for discounted future reward!
SAFETY_GAMMA = 0.9 # γ discount for penaltys only


#### SETTINGS FOR ACTOR #####

# Hyper Parameters
A_LAYER1_SIZE = 300
A_LAYER2_SIZE = 400
A_LEARNING_RATE = 1e-4
A_TAU = 0.001

##### SETTINGS FOR CRITC #####

C_LAYER1_SIZE = 300
C_LAYER2_SIZE = 600
C_LEARNING_RATE = 1e-3
C_TAU = 0.001
C_L2 = 0.0001