import tensorflow as tf
import numpy as np
import math
import config

LAYER1_SIZE = config.C_LAYER1_SIZE
LAYER2_SIZE = config.C_LAYER2_SIZE
LEARNING_RATE = config.C_LEARNING_RATE
TAU = config.C_TAU
L2 = config.C_L2


class Critic:
    """docstring for CriticNetwork"""

    def __init__(self, sess, state_dim, action_dim, num_actor_vars):
        self.time_step = 0

        self.sess = sess
        self.state_dim_sens = state_dim
        self.state_dim_vision = [64, 64, 3]
        self.action_dim = action_dim

        # create q network
        self.state_input_sens, \
        self.state_input_vision,\
        self.action_input, \
        self.q_value_output, \
        self.net = self.create_q_network(self.state_dim_sens,self.state_dim_vision, action_dim, "_critic")
        print("Critic network = " + str(self.net))

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # create target q network (the same structure with q network)
        self.state_input_sens_target, \
        self.state_input_vision_target, \
        self.target_action_input, \
        self.target_q_value_output, \
        self.target_net = self.create_q_network(self.state_dim_sens,self.state_dim_vision, action_dim, "_critic_target")
        print("Critic target network = " + str(self.target_net))

        self.target_network_params = tf.trainable_variables()[(num_actor_vars + len(self.network_params)):]

        # this is from the guide i used when reusing create network
        # TODO eclude conv layers??
        self.target_update = \
            [self.target_network_params[i].assign(
                tf.multiply(self.network_params[i], TAU) + tf.multiply(self.target_network_params[i], 1. - TAU))
                for i in range(len(self.target_network_params))]

        # create target q network (the same structure with q network)
        """self.target_state_input, \
        self.target_action_input, \
        self.target_q_value_output, \
        self.target_update, \
        self.target_net = self.create_target_q_network(state_dim, action_dim, self.net)
        print("Critic target_network = " + str(self.target_net))"""

        self.create_training_method()

        # initialization
        self.sess.run(tf.initialize_all_variables())

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

        self.update_target()

    def create_training_method(self):
        # Define training optimizer
        self.y_input = tf.placeholder("float", [None, 1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay

        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

        self.action_gradients = tf.gradients(self.q_value_output, self.action_input)

    def create_q_network(self, state_dim_sens, state_dim_vision, action_dim, name):
        # the layer size could be changed
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        ## This is just coppied from the conv of the actor... so make sure they stay identical
        # input for image
        state_input_vision = tf.placeholder(dtype="float", shape=([None] + state_dim_vision),
                                            name="state_input_vision" + name)

        # conv layer 1
        conv1 = tf.layers.conv2d(
            inputs=state_input_vision,
            filters=68,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            name="conv1" + name)

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name="pool1" + name)

        # conv layer 2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=136,  # assumed double 68, since guide did double 32
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            name="conv2" + name)

        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="pool2" + name)

        # Flatten last pool
        flat_size = 16 * 16 * 136
        pool2_flat = tf.reshape(pool2, [-1, flat_size])

        ## Fully Connected layers!
        # Input for sensors, not part of conv. net
        state_input_sens = tf.placeholder("float", [None, state_dim_sens])
        action_input = tf.placeholder("float", [None, action_dim])

        # Layer 1
        W1_sens = tf.Variable(tf.random_uniform([state_dim_sens, layer1_size], -1 / math.sqrt(state_dim_sens + flat_size), 1 / math.sqrt(state_dim_sens + flat_size)), name="W1_sens"+ name)
        W1_vision = tf.Variable(tf.random_uniform([flat_size, layer1_size], -1 / math.sqrt(state_dim_sens + flat_size), 1 / math.sqrt(state_dim_sens + flat_size)), name="W1_vision" + name)
        b1 = tf.Variable(tf.random_uniform([layer1_size], -1 / math.sqrt(state_dim_sens), 1 / math.sqrt(state_dim_sens)), name="b1"+ name)

        # Layer 2
        W2 = tf.Variable(tf.random_uniform([layer1_size, layer2_size], -1 / math.sqrt(layer1_size + action_dim), 1 / math.sqrt(layer1_size + action_dim)), name="W2"+ name)
        W2_action = tf.Variable(tf.random_uniform([action_dim, layer2_size], -1 / math.sqrt(layer1_size + action_dim), 1 / math.sqrt(layer1_size + action_dim)), name="W2_action"+ name)
        b2 = tf.Variable(tf.random_uniform([layer2_size], -1 / math.sqrt(layer1_size), 1 / math.sqrt(layer1_size)), name="b2"+ name)

        # Layer 3
        W3 = tf.Variable(tf.random_uniform([layer2_size, 1], -3e-3, 3e-3), name="W3"+ name)
        b3 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3), name="b3"+ name)

        layer1 = tf.nn.relu(tf.matmul(state_input_sens, W1_sens) + tf.matmul(pool2_flat, W1_vision) + b1, name="layer1"+ name)
        layer2 = tf.nn.relu(tf.matmul(layer1, W2) + tf.matmul(action_input, W2_action) + b2, name="layer2"+ name)
        q_value_output = tf.identity(tf.matmul(layer2, W3) + b3, name="q_value_output"+ name)

        return state_input_sens, state_input_vision, action_input, q_value_output, [conv1, pool1, conv2, pool2, pool2_flat, W1_sens, b1, W2, W2_action, b2, W3, b3]

    """def create_target_q_network(self, state_dim, action_dim, net):
        state_input = tf.placeholder("float", [None, state_dim])
        action_input = tf.placeholder("float", [None, action_dim])

        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1, target_net[2]) + tf.matmul(action_input, target_net[3]) + target_net[4])
        q_value_output = tf.identity(tf.matmul(layer2, target_net[5]) + target_net[6])

        return state_input, action_input, q_value_output, target_update, target_net"""

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, y_batch, state_batch, action_batch):
        self.time_step += 1
        state_batch_sens = np.asarray([data[0] for data in state_batch])
        state_batch_vision = np.asarray([data[1] for data in state_batch])
        self.sess.run(self.optimizer, feed_dict={
            self.y_input: y_batch,
            self.state_input_sens: state_batch_sens,
            self.state_input_vision: state_batch_vision,
            self.action_input: action_batch
        })

    def gradients(self, state_batch, action_batch):
        state_batch_sens = np.asarray([data[0] for data in state_batch])
        state_batch_vision = np.asarray([data[1] for data in state_batch])
        return self.sess.run(self.action_gradients, feed_dict={
            self.state_input_sens: state_batch_sens,
            self.state_input_vision: state_batch_vision,
            self.action_input: action_batch
        })[0]

    def target_q(self, state_batch, action_batch):
        state_batch_sens = np.asarray([data[0] for data in state_batch])
        state_batch_vision = np.asarray([data[1] for data in state_batch])
        return self.sess.run(self.target_q_value_output, feed_dict={
            self.state_input_sens_target: state_batch_sens,
            self.state_input_vision_target: state_batch_vision,
            self.target_action_input: action_batch
        })

    def q_value(self, state_batch, action_batch):
        state_batch_sens = np.asarray([data[0] for data in state_batch])
        state_batch_vision = np.asarray([data[1] for data in state_batch])
        return self.sess.run(self.q_value_output, feed_dict={
            self.state_input_sens: state_batch_sens,
            self.state_input_vision: state_batch_vision,
            self.action_input: action_batch})

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    # f fan-in size
    #def variable(self, shape, f):
    #    return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))

