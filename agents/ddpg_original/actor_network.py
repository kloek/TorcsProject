import tensorflow as tf
import numpy as np
import math

# Hyper Parameters
LAYER1_SIZE = 300
LAYER2_SIZE = 400
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 32


class Actor:
    """docstring for ActorNetwork"""

    def __init__(self, sess, state_dim, action_dim):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim

        # create actor network
        self.state_input, self.action_output, self.net = self.create_network(state_dim, action_dim)

        # create target actor network
        self.target_state_input, self.target_action_output, self.target_update, self.target_net = self.create_target_network(
            state_dim, action_dim, self.net)

        # define training rules
        self.create_training_method()

        self.sess.run(tf.initialize_all_variables())

        self.update_target()
        # self.load_network()

    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output, self.net, -self.q_gradient_input)
        '''        
        for i, grad in enumerate(self.parameters_gradients):
            if grad is not None:
                self.parameters_gradients[i] = tf.clip_by_value(grad, -2.0,2.0)
        '''
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients, self.net))

    def create_network(self, state_dim, action_dim):
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        state_input = tf.placeholder("float", [None, state_dim], name="state_input")

        #W1 = self.variable([state_dim, layer1_size], state_dim)
        W1 = tf.Variable(tf.random_uniform([state_dim, layer1_size], -1 / math.sqrt(state_dim), 1 / math.sqrt(state_dim)), name="W1")
        #b1 = self.variable([layer1_size], state_dim)
        b1 = tf.Variable(tf.random_uniform([layer1_size], -1 / math.sqrt(state_dim), 1 / math.sqrt(state_dim)), name="b1")

        #W2 = self.variable([layer1_size, layer2_size], layer1_size)
        W2 = tf.Variable(tf.random_uniform([layer1_size, layer2_size], -1 / math.sqrt(layer1_size), 1 / math.sqrt(layer1_size)), name="W2")
        #b2 = self.variable([layer2_size], layer1_size)
        b2 = tf.Variable(tf.random_uniform([layer2_size], -1 / math.sqrt(layer1_size), 1 / math.sqrt(layer1_size)), name="b2")

        # W3 = tf.Variable(tf.random_uniform([layer2_size,action_dim],-3e-3,3e-3))
        # b3 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))

        W_steer = tf.Variable(tf.random_uniform([layer2_size, 1], -1e-4, 1e-4), name="W_steer")
        b_steer = tf.Variable(tf.random_uniform([1], -1e-4, 1e-4), name="b_steer")

        W_accel = tf.Variable(tf.random_uniform([layer2_size, 1], -1e-4, 1e-4), name="W_accel")
        b_accel = tf.Variable(tf.random_uniform([1], -1e-4, 1e-4), name="b_accel")

        W_brake = tf.Variable(tf.random_uniform([layer2_size, 1], -1e-4, 1e-4), name="W_brake")
        b_brake = tf.Variable(tf.random_uniform([1], -1e-4, 1e-4), name="b_brake")

        layer1 = tf.nn.relu(tf.matmul(state_input, W1) + b1, name="layer1")
        layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2, name="layer2")

        steer = tf.tanh(tf.matmul(layer2, W_steer) + b_steer, name="steer")
        accel = tf.sigmoid(tf.matmul(layer2, W_accel) + b_accel, name="accel")
        brake = tf.sigmoid(tf.matmul(layer2, W_brake) + b_brake, name="brake")

        action_output = tf.concat([steer, accel, brake], 1, name="action_output")

        return state_input, action_output, [W1, b1, W2, b2, W_steer, b_steer, W_accel, b_accel, W_brake, b_brake]

    def create_target_network(self, state_dim, action_dim, net):
        state_input = tf.placeholder("float", [None, state_dim])
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1, target_net[2]) + target_net[3])

        steer = tf.tanh(tf.matmul(layer2, target_net[4]) + target_net[5])
        accel = tf.sigmoid(tf.matmul(layer2, target_net[6]) + target_net[7])
        brake = tf.sigmoid(tf.matmul(layer2, target_net[8]) + target_net[9])

        action_output = tf.concat([steer, accel, brake], 1)
        return state_input, action_output, target_update, target_net

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch
        })

    def actions(self, state_batch):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: state_batch
        })

    def action(self, state):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: [state]
        })[0]

    def target_actions(self, state_batch):
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_state_input: state_batch
        })

    # f fan-in size
    def variable(self, shape, f):
        return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))


    def print_settings(self, settings_file):
        settings_text = ["\n\n==== from actor network ====" + "\n",
                         "LAYER1_SIZE = " + str(LAYER1_SIZE) + "\n",
                         "LAYER2_SIZE = " + str(LAYER2_SIZE) + "\n",
                         "LEARNING_RATE = " + str(LEARNING_RATE) + "\n",
                         "TAU = " + str(TAU) + "\n",
                         "BATCH_SIZE = " + str(BATCH_SIZE) + "\n"]
        for line in settings_text:
            settings_file.write(line)  # print settings to file
