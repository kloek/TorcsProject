# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from agents.abstract_agent import AbstractAgent
from agents.ddpg_original.actor_network import Actor
from agents.ddpg_original.critic_network import Critic
import config

from agents.parts.OU import OU
from agents.parts.replay_buffer import ReplayBuffer


# Notes for readability: comments with tripple sharp (###) is the main steps of ddpg algorithm
# ddpg from : https://arxiv.org/pdf/1509.02971.pdf

class Agent(AbstractAgent):



    # Hyper Parameters:
    REPLAY_BUFFER_SIZE = config.REPLAY_BUFFER_SIZE
    REPLAY_START_SIZE = config.REPLAY_START_SIZE
    BATCH_SIZE = config.BATCH_SIZE
    GAMMA = config.GAMMA

    actor_network = None
    critic_network = None
    replay_buffer = None
    OU = None

    #Construct ddpg agent
    def __init__(self, env_name, state_dim, action_dim, safety_critic=False):
        self.state_dim = state_dim  # dimention of states e.g vision size and other sensors!
        self.action_dim = action_dim  # dimention of action e.g 3 for steering, throttle, and break
        self.safety_critic = safety_critic  # false = normal ddpg, true = safety critic test!

        self.env_name = env_name

        # Ensure action bound is symmetric
        self.time_step = 0
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))

        ### Randomly initialize critic network and actor with weights θQ and θμ
        print("creating actor and critic")
        self.actor_network = Actor(self.sess, self.state_dim, self.action_dim)
        self.critic_network = Critic(self.sess, self.state_dim, self.action_dim)

        # create an extra safety critic:
        if (self.safety_critic):
            self.safety_critic_network = Critic(self.sess, self.state_dim, self.action_dim)

        ### Initialize replay buffer R
        self.replay_buffer = ReplayBuffer(self.REPLAY_BUFFER_SIZE)

        ### Initialize a random process for action exploration (Ornstein-Uhlenbeck process)
        self.OU = OU()

        # loading networks #TODO, this is not adapted to my code!!!
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_networks/")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def act(self, s_t, is_training, epsilon ,done):
        ## create action based on observed state s_t
        #TODO not adapted to diffrent action dims!!!!

        action = self.actor_network.action(s_t)

        if(is_training):
            # OU function(self, x, mu, theta, sigma)
            noise_t = np.zeros(self.action_dim)
            noise_t[0] = epsilon * self.OU.function(action[0], 0.0, 0.60, 0.80)
            noise_t[1] = epsilon * self.OU.function(action[1], 0.5, 1.00, 0.10)
            noise_t[2] = epsilon * self.OU.function(action[2], -0.1, 1.00, 0.05)
            action = action + noise_t


        action[0] = np.clip(action[0], -1, 1)
        action[1] = np.clip(action[1], 0, 1)
        action[2] = np.clip(action[2], 0, 1)

        return action

    def train(self):

        ### Sample a random minibatch of N (BATCH_SIZE) transitions (s_i, a_i, r_i, s_i+1) from ReplayBuffer
        # print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.getBatch(self.BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])

        # the reward batch is no longer just one column! (it's a list of 4 cols....)
        reward_batch = np.asarray([data[2] for data in minibatch])

        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch, [self.BATCH_SIZE, self.action_dim])

        # Calculate y_batch and 
        # Update critic by minimizing the loss L
        if(self.safety_critic):
            y_batch_progress = self.calc_y_batch(done_batch, minibatch, next_state_batch, reward_batch, 1, gamma=self.GAMMA)
            y_batch_penalty = self.calc_y_batch(done_batch, minibatch, next_state_batch, reward_batch, 2, gamma=self.GAMMA)
            self.critic_network.train(y_batch_progress, state_batch, action_batch)
            self.safety_critic_network.train(y_batch_penalty, state_batch, action_batch)
        else:
            y_batch_reward = self.calc_y_batch(done_batch, minibatch, next_state_batch, reward_batch, 0, gamma=self.GAMMA)
            self.critic_network.train(y_batch_reward, state_batch, action_batch)

        ## Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        if(self.safety_critic):
            q_gradient_batch_progress = self.critic_network.gradients(state_batch, action_batch_for_gradients)
            q_gradient_batch_penalty = self.safety_critic_network.gradients(state_batch, action_batch_for_gradients)
            self.actor_network.train(q_gradient_batch_progress, state_batch)
            self.actor_network.train(q_gradient_batch_penalty, state_batch)
        else:
            q_gradient_batch = self.critic_network.gradients(state_batch, action_batch_for_gradients)
            self.actor_network.train(q_gradient_batch, state_batch)

        #self.actor_network.train(q_gradient_batch, state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()
        if(self.safety_critic):
            self.safety_critic_network.update_target()

    def calc_y_batch(self, done_batch, minibatch, next_state_batch, reward_batch, reward_col, gamma):
        # next_action = μ'(st+1 | θ'μ')
        next_action_batch = self.actor_network.target_actions(next_state_batch)
        # Q_values = Q'(Si+1, next_action | θ'Q)
        q_value_batch = self.critic_network.target_q(next_state_batch, next_action_batch)
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i, reward_col])
            else:
                y_batch.append(reward_batch[i, reward_col] + gamma * q_value_batch[i])
        y_batch = np.resize(y_batch, [self.BATCH_SIZE, 1])
        return y_batch

    @staticmethod
    def get_name():
        return "DDPG Original"


    def save_networks(self, global_step, run_folder):
        self.saver.save(self.sess, run_folder+'/saved_networks/' + self.env_name + 'network' + '-ddpg', global_step=global_step)
        #self.actor_network.save_network(global_step=global_step, run_folder=run_folder)
        #self.critic_network.save_network(global_step=global_step, run_folder=run_folder)
