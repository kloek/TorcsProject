import tensorflow as tf
import numpy as np
import math
import config

# Hyper Parameters
LAYER1_SIZE = config.A_LAYER1_SIZE
LAYER2_SIZE = config.A_LAYER2_SIZE
LEARNING_RATE = config.A_LEARNING_RATE
TAU = config.A_TAU


class Actor:
    """docstring for ActorNetwork"""

    def __init__(self, sess, state_dim, action_dim):
        self.sess = sess
        self.state_dim_sens = state_dim
        self.state_dim_vision = [64,64,3]
        self.action_dim = action_dim

        # create actor network
        self.state_input_sens, \
        self.state_input_vision, \
        self.action_output, \
        self.net = self.create_network(self.state_dim_sens, self.state_dim_vision, action_dim, "_actor")
        print("Actor network = " + str(self.net))

        self.network_params = tf.trainable_variables()
        print("network_params = " + str(self.network_params))

        # create actor target network
        self.state_input_sens_target, \
        self.state_input_vision_target, \
        self.target_action_output, \
        self.target_net = self.create_network(self.state_dim_sens, self.state_dim_vision, action_dim, "_actor_target")
        print("Actor target network = " + str(self.target_net))

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]
        print("target_network_params = " + str(self.target_network_params))

        ## create target update!
        #ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        #self.target_update = ema.apply(self.net)
        #target_net = [ema.average(x) for x in net]

        # this is from the guide i used when reusing create network
        #TODO eclude conv layers??
        self.target_update = \
            [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], TAU) + tf.multiply(self.target_network_params[i], 1. - TAU))
             for i in range(len(self.target_network_params))]

        #target_net = [ema.average(x) for x in net]

        ## create target actor network
        #self.target_state_input, \
        #self.target_action_output, self.target_update, self.target_net = self.create_target_network(
        #    state_dim, action_dim, self.net, self.conv_net)
        #print("Actor target_network = " + str(self.target_net))

        self.sess.run(tf.initialize_all_variables())

        # define training rules
        self.create_training_method()

        self.update_target()
        # self.load_network()

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    # when implementing cnns, self.net had to be changed to self.network_params!!!
    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output, self.network_params, -self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients, self.network_params))

    def create_network(self, state_dim_sens, state_dim_vision, action_dim, name):
        print(" === Create Network ("+name+") === ")

        #fully connected layers
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        # input for image
        state_input_vision = tf.placeholder(dtype="float", shape=([None] + state_dim_vision), name="state_input_vision"+name)

        #conv layer 1
        conv1 = tf.layers.conv2d(
            inputs=state_input_vision,
            filters=68,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu,
            name="conv1"+name)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2, name="pool1"+name)

        # conv layer 2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=136,  # assumed double 68, since guide did double 32
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            name="conv2"+name)

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="pool2"+name)

        # Flatten last pool
        flat_size = 16*16*136
        pool2_flat = tf.reshape(pool2, [-1,flat_size])


        ## Fully Connected layers!
        # Input for sensors, not part of conv. net
        state_input_sens = tf.placeholder("float", [None, state_dim_sens], name="state_input_sens" + name)

        # Layer 1
        W1_sens = tf.Variable(tf.random_uniform([state_dim_sens, layer1_size], -1 / math.sqrt(state_dim_sens + flat_size), 1 / math.sqrt(state_dim_sens + flat_size)), name="W1_sens"+name)
        W1_vision = tf.Variable(tf.random_uniform([flat_size, layer1_size], -1 / math.sqrt(state_dim_sens + flat_size), 1 / math.sqrt(state_dim_sens + flat_size)), name="W1_vision"+name)
        b1 = tf.Variable(tf.random_uniform([layer1_size], -1 / math.sqrt(state_dim_sens), 1 / math.sqrt(state_dim_sens)), name="b1"+name)

        # Layer 2
        W2 = tf.Variable(tf.random_uniform([layer1_size, layer2_size], -1 / math.sqrt(layer1_size), 1 / math.sqrt(layer1_size)), name="W2"+name)
        b2 = tf.Variable(tf.random_uniform([layer2_size], -1 / math.sqrt(layer1_size), 1 / math.sqrt(layer1_size)), name="b2"+name)

        # W3 = tf.Variable(tf.random_uniform([layer2_size,action_dim],-3e-3,3e-3))
        # b3 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))

        W_steer = tf.Variable(tf.random_uniform([layer2_size, 1], -1e-4, 1e-4), name="W_steer"+name)
        b_steer = tf.Variable(tf.random_uniform([1], -1e-4, 1e-4), name="b_steer"+name)

        W_accel = tf.Variable(tf.random_uniform([layer2_size, 1], -1e-4, 1e-4), name="W_accel"+name)
        b_accel = tf.Variable(tf.random_uniform([1], -1e-4, 1e-4), name="b_accel"+name)

        W_brake = tf.Variable(tf.random_uniform([layer2_size, 1], -1e-4, 1e-4), name="W_brake"+name)
        b_brake = tf.Variable(tf.random_uniform([1], -1e-4, 1e-4), name="b_brake"+name)

        layer1 = tf.nn.relu(tf.matmul(state_input_sens, W1_sens) + tf.matmul(pool2_flat, W1_vision) + b1, name="layer1"+name)
        layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2, name="layer2"+name)

        steer = tf.tanh(tf.matmul(layer2, W_steer) + b_steer, name="steer"+name)
        accel = tf.sigmoid(tf.matmul(layer2, W_accel) + b_accel, name="accel"+name)
        brake = tf.sigmoid(tf.matmul(layer2, W_brake) + b_brake, name="brake"+name)
#conv1, pool1, conv2, pool2, pool2_flat,
        action_output = tf.concat([steer, accel, brake], 1, name="action_output"+name)
        return state_input_sens, state_input_vision, action_output, [conv1, pool1, conv2, pool2, pool2_flat, W1_sens, b1, W2, b2, W_steer, b_steer, W_accel, b_accel, W_brake, b_brake]

    """def create_target_network(self, state_dim, action_dim, net, conv_net):
        print(" === Create Target Network (Actor) === ")

        state_input = tf.placeholder("float", [None, state_dim])
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        img_input = tf.placeholder(dtype="float", shape=[None, 64, 64, 3], name="image_input")
        ema_conv = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update_conv = ema.apply(conv_net)
        target_conv_net = [ema.average(x) for x in conv_net]

        print("net = " + str(net))
        print("target_net = " + str(target_net))
        print("conv_net = " + str(conv_net))

        layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1, target_net[2]) + target_net[3])

        steer = tf.tanh(tf.matmul(layer2, target_net[4]) + target_net[5])
        accel = tf.sigmoid(tf.matmul(layer2, target_net[6]) + target_net[7])
        brake = tf.sigmoid(tf.matmul(layer2, target_net[8]) + target_net[9])

        action_output = tf.concat([steer, accel, brake], 1)
        return state_input, action_output, target_update, target_net"""

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, q_gradient_batch, state_batch):
        state_batch_sens = state_batch[0]
        state_batch_vision = state_batch[1]
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input_sens: state_batch_sens,
            self.state_input_vision: state_batch_vision
        })

    def actions(self, state_batch):
        state_batch_sens = state_batch[0]
        state_batch_vision = state_batch[1]
        return self.sess.run(self.action_output, feed_dict={
            self.state_input_sens: state_batch_sens,
            self.state_input_vision: state_batch_vision
        })

    def action(self, state):
        state_sens = state[0]
        state_vision = state[1]
        return self.sess.run(self.action_output, feed_dict={
            self.state_input_sens: [state_sens],
            self.state_input_vision: [state_vision]
        })[0]

    def target_actions(self, state_batch):
        state_batch_sens = state_batch[0]
        state_batch_vision = state_batch[1]
        return self.sess.run(self.target_action_output, feed_dict={
            self.state_input_sens_target: state_batch_sens,
            self.state_input_vision_target: state_batch_vision
        })

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    # f fan-in size
    #def variable(self, shape, f):
    #    return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))
