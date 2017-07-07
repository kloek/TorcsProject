# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math

# Hyper Parameters
LAYER1_SIZE = 300
LAYER2_SIZE = 600
LEARNING_RATE = 1e-3
TAU = 1e-3
L2 = 1e-4

class Critic(object):


    def __init__(self, session, state_dim, action_dim):
        self.time_step = 0
        self.session = session
        self.state_dim = state_dim
        self.action_dim = action_dim


        ### Initialize target network Q′ with weights θQ′ ← θQ
        # create q network
        self.state_input, \
        self.action_input, \
        self.q_value_output, \
        self.net = self.create_q_network(state_dim, action_dim)

        # create target q network
        self.target_state_input, \
        self.target_action_input, \
        self.target_q_value_output, \
        self.target_update = self.create_target_q_network(state_dim, action_dim, self.net)

        # define training rules
        self.create_training_method()

        # initialization random weights, (Q′ with weights θQ′ ← θQ)
        init = tf.global_variables_initializer();
        self.session.run(init)

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

        W1_shape = [state_dim, layer1_size]
        W1 = tf.Variable(tf.random_uniform(W1_shape, -1 / math.sqrt(state_dim), 1 / math.sqrt(state_dim)))
        b1_shape = [layer1_size]
        b1 = tf.Variable(tf.random_uniform(b1_shape, -1 / math.sqrt(state_dim), 1 / math.sqrt(state_dim)))


        W2_shape = [layer1_size, layer2_size]
        W2 = tf.Variable(tf.random_uniform(W2_shape, -1 / math.sqrt(layer1_size+action_dim), 1 / math.sqrt(layer1_size+action_dim)))
        W2_action = tf.Variable(tf.random_uniform([action_dim, layer2_size], -1 / math.sqrt(layer1_size+action_dim), 1 / math.sqrt(layer1_size+action_dim)))
        b2_shape = [layer2_size]
        b2 = tf.Variable(tf.random_uniform(b2_shape, -1 / math.sqrt(layer1_size+action_dim), 1 / math.sqrt(layer1_size+action_dim)))

        W3 = tf.Variable(tf.random_uniform([layer2_size, 1], -3e-3, 3e-3))
        b3 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))


        layer1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, W2) + tf.matmul(action_input, W2_action) + b2)
        q_value_output = tf.identity(tf.matmul(layer2, W3) + b3)

        return state_input, action_input, q_value_output, [W1, b1, W2, W2_action, b2, W3, b3]

    #TODO, could original "create_q_network" be used for both?
    def create_target_q_network(self, state_dim, action_dim, net):
        state_input = tf.placeholder("float", [None, state_dim])
        action_input = tf.placeholder("float", [None, action_dim])

        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1, target_net[2]) + tf.matmul(action_input, target_net[3]) + target_net[4])
        q_value_output = tf.identity(tf.matmul(layer2, target_net[5]) + target_net[6])

        return state_input, action_input, q_value_output, target_update

    # TODO taken from ddgp torcs tensorflow... (all below!) check how they work!!!!!
    def update_target(self):
        self.session.run(self.target_update)

    def train(self, y_batch, state_batch, action_batch):
        self.time_step += 1
        self.session.run(self.optimizer, feed_dict={
            self.y_input: y_batch,
            self.state_input: state_batch,
            self.action_input: action_batch
        })

    def gradients(self, state_batch, action_batch):
        return self.session.run(self.action_gradients, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch
        })[0]

    def target_q(self, state_batch, action_batch):
        return self.session.run(self.target_q_value_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_action_input: action_batch
        })

    def q_value(self, state_batch, action_batch):
        return self.session.run(self.q_value_output, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch
        })