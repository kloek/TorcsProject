# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from agents.abstract_agent import AbstractAgent
from agents.ddpg_original.actor_network import Actor
from agents.ddpg_original.critic_network import Critic

from agents.parts.OU import OU
from agents.parts.replay_buffer import ReplayBuffer


# Notes for readability: comments with tripple sharp (###) is the main steps of ddpg algorithm
# ddpg from : https://arxiv.org/pdf/1509.02971.pdf




class Agent(AbstractAgent):

    AGENT_NAME = "DDPG Original"

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
    def __init__(self,env_name, state_dim, action_dim):
        self.state_dim = state_dim  # dimention of states e.g vision size and other sensors!
        self.action_dim = action_dim  # dimention of action e.g 3 for steering, throttle, and break

        self.env_name = env_name

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

        # loading networks
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
        batch, batch_size = self.replay_buffer.getBatch(self.BATCH_SIZE) # returns batch_size since it can be smaller than self.BATCH_SIZE
        state_batch = np.asarray([data[0] for data in batch])
        action_batch = np.asarray([data[1] for data in batch])
        reward_batch = np.asarray([data[2] for data in batch])
        next_state_batch = np.asarray([data[3] for data in batch])
        done_batch = np.asarray([data[4] for data in batch])

        ### Set yi = ri + γQ′(si+1,μ′(si+1|θμ′)|θQ′)
        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch, next_action_batch)
        y_batch = []
        for i in range(batch_size):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.GAMMA * q_value_batch[i])

        y_batch = np.resize(y_batch, [batch_size, 1])

        ### Update critic by minimizing the loss:
        self.critic_network.train(y_batch, state_batch, action_batch)

        ### Update the actor policy using the sampled policy gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch, action_batch_for_gradients)
        self.actor_network.train(q_gradient_batch, state_batch)

        ### Update the target networks:
        self.actor_network.update_target()
        self.critic_network.update_target()

    def get_name(self):
        return self.AGENT_NAME

    def print_settings(self, settings_file):
        # 1. print settings of this agent
        settings_text = ["\n\n==== from agent ====" + "\n",
                         "REPLAY_BUFFER_SIZE = " + str(self.REPLAY_BUFFER_SIZE) + "\n",
                         "REPLAY_START_SIZE = " + str(self.REPLAY_START_SIZE) + "\n",
                         "BATCH_SIZE = " + str(self.BATCH_SIZE) + "\n",
                         "GAMMA = " + str(self.GAMMA) + "\n"]
        for line in settings_text:
            settings_file.write(line)  # print settings to file

        # 2. print settings of actor
        self.actor_network.print_settings(settings_file)

        # 3. print settings of critic
        self.critic_network.print_settings(settings_file)

    # TODO!!!!!!
    def save_results(self):
        print("")

    def save_networks(self, global_step, run_folder):
        self.saver.save(self.sess, run_folder+'/saved_networks/' + self.env_name + 'network' + '-ddpg', global_step=global_step)
        #self.actor_network.save_network(global_step=global_step, run_folder=run_folder)
        #self.critic_network.save_network(global_step=global_step, run_folder=run_folder)
