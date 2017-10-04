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
        # create q network
        self.state_input, \
        self.action_input, \
        self.q_value_output, \
        self.net = self.create_q_network(state_dim, action_dim)
        print("Critic network = " + str(self.net))

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # create target q network (the same structure with q network)
        self.target_state_input, \
        self.target_action_input, \
        self.target_q_value_output, \
        self.target_update, \
        self.target_net = self.create_target_q_network(state_dim, action_dim, self.net)
        print("Critic target_network = " + str(self.target_net))

        self.target_network_params = tf.trainable_variables()[(num_actor_vars + len(self.network_params)):]

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

    def create_q_network(self, state_dim, action_dim):
        # the layer size could be changed
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        state_input = tf.placeholder("float", [None, state_dim])
        action_input = tf.placeholder("float", [None, action_dim])

        # W1 = self.variable([state_dim, layer1_size], state_dim)
        W1 = tf.Variable(tf.random_uniform([state_dim, layer1_size], -1 / math.sqrt(state_dim), 1 / math.sqrt(state_dim)), name="W1")
        # b1 = self.variable([layer1_size], state_dim)
        b1 = tf.Variable(tf.random_uniform([layer1_size], -1 / math.sqrt(state_dim), 1 / math.sqrt(state_dim)), name="b1")

        # W2 = self.variable([layer1_size, layer2_size], layer1_size)
        W2 = tf.Variable(tf.random_uniform([layer1_size, layer2_size], -1 / math.sqrt(layer1_size + action_dim), 1 / math.sqrt(layer1_size + action_dim)), name="W2")
        # W2_action = self.variable([action_dim, layer2_size], layer1_size + action_dim)
        W2_action = tf.Variable(tf.random_uniform([action_dim, layer2_size], -1 / math.sqrt(layer1_size + action_dim), 1 / math.sqrt(layer1_size + action_dim)), name="W2_action")
        # b2 = self.variable([layer2_size], layer1_size)
        b2 = tf.Variable(tf.random_uniform([layer2_size], -1 / math.sqrt(layer1_size), 1 / math.sqrt(layer1_size)), name="b2")

        W3 = tf.Variable(tf.random_uniform([layer2_size, 1], -3e-3, 3e-3), name="W3")
        b3 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3), name="b3")

        layer1 = tf.nn.relu(tf.matmul(state_input, W1) + b1, name="layer1")
        layer2 = tf.nn.relu(tf.matmul(layer1, W2) + tf.matmul(action_input, W2_action) + b2, name="layer2")
        q_value_output = tf.identity(tf.matmul(layer2, W3) + b3, name="q_value_output")

        return state_input, action_input, q_value_output, [W1, b1, W2, W2_action, b2, W3, b3]

    def create_target_q_network(self, state_dim, action_dim, net):
        state_input = tf.placeholder("float", [None, state_dim])
        action_input = tf.placeholder("float", [None, action_dim])

        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1, target_net[2]) + tf.matmul(action_input, target_net[3]) + target_net[4])
        q_value_output = tf.identity(tf.matmul(layer2, target_net[5]) + target_net[6])

        return state_input, action_input, q_value_output, target_update, target_net

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, y_batch, state_batch, action_batch):
        self.time_step += 1
        self.sess.run(self.optimizer, feed_dict={
            self.y_input: y_batch,
            self.state_input: state_batch,
            self.action_input: action_batch
        })

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch
        })[0]

    def target_q(self, state_batch, action_batch):
        return self.sess.run(self.target_q_value_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_action_input: action_batch
        })

    def q_value(self, state_batch, action_batch):
        return self.sess.run(self.q_value_output, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch})

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

