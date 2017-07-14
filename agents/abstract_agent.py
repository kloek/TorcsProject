
class AbstractAgent(object):

    def __init__(self, dim_action):
        raise NotImplementedError

    # Get an Observation (ob) from the environment.
    # Each observation vectors are numpy array.
    # focus, opponents, track sensors are scaled into [0, 1]. When the agent
    # is out of the road, sensor variables return -1/200.
    # rpm, wheelSpinVel are raw values and then needed to be preprocessed.
    # vision is given as a tensor with size of (64*64, 3) = (4096, 3) <-- rgb
    # and values are in [0, 255]
    def act(self, s_t, is_training, epsilon ,done):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    # name of agent
    def get_name(self):
        raise NotImplementedError

    # print settings to settings file
    def print_settings(self, settings_file):
        raise NotImplementedError

    def save_results(self):
        raise NotImplementedError

    def save_networks(self):
        raise NotImplementedError
