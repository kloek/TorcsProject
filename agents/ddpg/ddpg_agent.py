# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from agents.abstract_agent import AbstractAgent
from agents.ddpg.OU import OU
from agents.ddpg.actor_network import Actor
from agents.ddpg.critic_network import Critic
from agents.ddpg.replay_buffer import ReplayBuffer


# Notes for readability: comments with tripple sharp (###) is the main steps of ddpg algorithm
# ddpg from : https://arxiv.org/pdf/1509.02971.pdf




class Agent(AbstractAgent):

    # Hyper Parameters:
    REPLAY_BUFFER_SIZE = 100000
    REPLAY_START_SIZE = 100
    BATCH_SIZE = 32 # size of minibatches to train with
    GAMMA = 0.99    # γ discount factor for discounted future reward!

    actor_network = None
    critic_network = None
    replay_buffer = None
    OU = None

    #Construct ddpg agent
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim  # dimention of states e.g vision size and other sensors!
        self.action_dim = action_dim  # dimention of action e.g 3 for steering, throttle, and break

        # Ensure action bound is symmetric
        self.time_step = 0
        self.sess = tf.InteractiveSession()

        ### Randomly initialize critic network and actor with weights θQ and θμ
        self.actor_network = Actor(self.sess, self.state_dim, self.action_dim)
        self.critic_network = Critic(self.sess, self.state_dim, self.action_dim)

        ### Initialize replay buffer R
        self.replay_buffer = ReplayBuffer(self.REPLAY_BUFFER_SIZE)

        ### Initialize a random process for action exploration (Ornstein-Uhlenbeck process)
        self.OU = OU()

    def act(self, s_t, is_training ,done):
        ## create action based on observed state s_t
        #TODO not adapted to diffrent action dims!!!!
        # print("s_t = " + str(s_t))
        action = self.actor_network.action(s_t)
        action[0] = np.clip(action[0], -1, 1)
        action[1] = np.clip(action[1], 0, 1)
        action[2] = np.clip(action[2], 0, 1)
        # print "Action:", action
        return action

    def train(self):

        ### Sample a random minibatch of N (BATCH_SIZE) transitions (s_i, a_i, r_i, s_i+1) from ReplayBuffer
        batch = self.replay_buffer.getBatch(self.BATCH_SIZE)

        ### Set yi = ri + γQ′(si+1,μ′(si+1|θμ′)|θQ′)
        # TODO

        ### Update critic by minimizing the loss:
        # TODO

        ### Update the actor policy using the sampled policy gradient:
        # TODO

        ### Update the target networks:
        # TODO

    def get_name(self):
        return "DDPG Agent"
