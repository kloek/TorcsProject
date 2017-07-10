# -*- coding: utf-8 -*-

from gym_torcs import TorcsEnv
from agents.ddpg.ddpg_agent import Agent
import numpy as np
import random
import math
import os
import datetime

# Notes for readability: comments with tripple sharp (###) is the steps of main algorithm
# these tripple comments are spread across multiple files since diffrent part of algo is performed by diffrent classes
# such as run_ddpg, which contains eppisode and steps loops... ddpg_agent which contains agent related functions, and so on....
# ddpg from : https://arxiv.org/pdf/1509.02971.pdf

class agent_runner(object):

    is_training = 1  # TODO sys arg or config file
    epsilon_start = 1.0  # TODO sys arg or config file
    episode_count = 1  # TODO sys arg or config file
    max_steps = 50  # TODO sys arg or config file

    best_reward = 0

    # initial values
    reward = 0
    done = False
    epsilon = epsilon_start

    # Gym_torcs
    vision = False
    throttle = True
    gear_change = False
    # brake = true #TODO

    # env and agent
    state_dim = 29  # TODO
    action_dim = 3

    env = None
    agent = None

    # for saving settings
    folder_name = None
    settings_file = None

    def __init__(self):
        # Generate a Torcs environment
        self.env = TorcsEnv(vision=self.vision, throttle=self.throttle, gear_change=self.gear_change)
        print("1. Env is created!")

        #TODO ALL THIS IS EXPERIMENTAL!!!!!
        #TODO, why is action state size two when it should be 3?
        #TODO, use this to set action_dim and state_dim
        #action_dim_test = env.action_space.sample().shape
        #print("action_dim_test = " + str(action_dim_test))
        #state_dim_test = env.observation_space.sample().shape
        #s_t = env.observation_space.sample()
        #print("State dim_test" + str(state_dim_test.shape))
        #print("sample does work")
        #print("action_dim=" + str(action_dim) + ",  state_dim=" + str(state_dim))

        self.agent = Agent(state_dim=self.state_dim, action_dim=self.action_dim)
        print("2. Agent is created!")

        # create a folder in runs for saving info about the run, result, and trained nets!!
        start_time = datetime.datetime.now()
        self.folder_name = "runs/" + start_time.strftime("%Y-%m-%d %H:%M:%S - " + self.agent.get_name())
        os.makedirs(self.folder_name)

        # create a settings file ( only for saving setting, not for applying settings!!!!
        self.settings_file = open(self.folder_name + "/" + "settings", "a")

        self.print_settings(settings_file=self.settings_file)  # print settings from runfile
        self.agent.print_settings(settings_file=self.settings_file)  # print settings from agent

    def run_ddpg(self):

        ### for episode = 1, M
        for episode in range(self.episode_count):
            print("=============================================================")
            print(" starting episode: " + str(episode) +"/"+ str(self.episode_count))
            done = self.done

            # start with short eppisodes then increase them!
            if(self.max_steps < 300):
                self.max_steps += 5

            # train_indicator is equal to is_training but set to false when testing every 20th episode!
            train_indicator = (self.is_training and not((episode > 10) and (episode % 20 == 0)))

            ### Initialize a random process N for action exploration
            #Done in ddpg_agent constructor... OU

            #TODO Early stop? - train indicator is not is_training, but wheter a test run is active or not!
            #early_stop = do_early_stop(epsilon, train_indicator)

            ### Receive initial observation state s_t
            # relaunch TORCS every 5 episode because of the memory leak error
            ob = self.env.reset(relaunch=((episode % 5) == 0))
            s_t = self.create_state(ob)


            ### for t = 1, T
            for step in range(self.max_steps):

                ### Execute action at and observe reward rt and observe new state st+1
                a_t = self.agent.act(s_t=s_t, is_training=self.is_training, epsilon=self.epsilon, done=done)

                # send that action to the environment
                ob, r_t, done, info = self.env.step(a_t)
                s_t1 = self.create_state(ob) # next state, after action a_t
                if self.best_reward < r_t:
                    self.best_reward = r_t

                ### Store transition (st,at,rt,st+1) in ReplayBuffer
                self.agent.replay_buffer.add(s_t, a_t, r_t, s_t1, done)
                s_t = s_t1

                ### training (includes 5 steps from ddpg algo):
                trainstr = ""
                if(train_indicator):
                    trainstr = ", is training and "
                    self.agent.train()

                print("step: " + str(step) + ",  a_t=" + str(a_t) + ", r_t=" + str(r_t) + "/"+ str(self.best_reward) + trainstr  + " done = " + str(done) )

                # so that this loop stops if torcs is restarting or done!
                if done:
                    break
            ### end for
        ### end for

        self.finish()


    def print_settings(self, settings_file):
        settings_text = ["==== from runfile ====" + "\n",
                        "is_training = " + str(self.is_training) + "\n",
                         "epsilon_start = " + str(self.epsilon_start) + "\n",
                         "episode_count = " + str(self.episode_count) + "\n",
                         "max_steps = " + str(self.max_steps) + "\n",
                         "vision = " + str(self.vision) + "\n",
                         "throttle = " + str(self.throttle) + "\n",
                         "gear_change = " + str(self.gear_change) + "\n"]
        for line in settings_text:
            settings_file.write(line)  # print settings to file


    def create_state(self, ob):
        # TODO this is without vision!!!!!
        """names = ['angle',
             'curLapTime',
             'damage',
             'distFromStart',
             'distRaced',
             'focus',
             'fuel',
             'gear',
             'lastLapTime',
             'opponents',
             'racePos',
             'rpm',
             'speedX',
             'speedY',
             'speedZ',
             'track',
             'trackPos',
             'wheelSpinVel',
             'z']"""

        # print("observation=" + str(ob))
        # some numbers are scaled, se scale_observation(..) in gym_torcs
        s_t = np.hstack((ob['angle'], ob['track'], ob['trackPos'], ob['speedX'], ob['speedY'], ob['speedZ'], ob['wheelSpinVel'], ob['rpm']))
        return s_t

    def do_early_stop(epsilon, train_indicator):
        random_number = random.random()
        eps_early = max(epsilon, 0.10)
        return (random_number < (1.0 - eps_early)) and (train_indicator == 1)

    def finish(self):
        # add finished to the run folder!
        os.system("mv " + self.folder_name.replace(" ", "\ ")  + " " + (self.folder_name+" FINISHED").replace(" ", "\ "))

        self.env.end()  # This is for shutting down TORCS

if __name__ == "__main__":
    runner = agent_runner()
    runner.run_ddpg()



