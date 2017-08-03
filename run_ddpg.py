# -*- coding: utf-8 -*-

from gym_torcs import TorcsEnv
from agents.ddpg_original.ddpg_agent import Agent
import numpy as np
import random
import math
import os
import sys
import datetime

import gc
gc.enable()

from agents.parts.results import results

#test
import matplotlib
import matplotlib.pyplot as plt

# Notes for readability: comments with tripple sharp (###) is the steps of main algorithm
# these tripple comments are spread across multiple files since diffrent part of algo is performed by diffrent classes
# such as run_ddpg, which contains eppisode and steps loops... ddpg_agent which contains agent related functions, and so on....
# ddpg from : https://arxiv.org/pdf/1509.02971.pdf

class agent_runner(object):

    is_training = True  # TODO sys arg or config file
    test_frequency = 20 # TODO sys arg or config file # how often to test /episodes
    epsilon_start = 1  # TODO sys arg or config file
    episode_count = 1000  # TODO sys arg or config file
    max_steps = 1000  # TODO sys arg or config file
    EXPLORE = 400000.0

    log_size = 100 # number of episodes per log

    # initial values
    best_training_reward = -math.inf  # best training reward over all episodes and steps
    best_testing_reward = -math.inf

    total_steps = 0 # total nr of steps over all episodes
    start_time = None
    done = False
    epsilon = epsilon_start

    # Gym_torcs
    vision = False
    throttle = True
    gear_change = False #False = drive only on first gear, limited to 80 km/h



    # 1. original sensors!!!
    #state_dim = 30

    # 2. realistic sensors!! (vithout vision)
    #state_dim = 89

    # 3. combo! for driving without vision, but close to realistic!
    state_dim = 90

    action_dim = 3

    env = None
    agent = None

    # for saving settings
    folder_name = None
    settings_file = None
    result = None


    def __init__(self):

        # create a folder in runs for saving info about the run, result, and trained nets!!
        self.start_time = datetime.datetime.now()
        self.folder_name = "runs/" + self.start_time.strftime("%Y-%m-%d %H:%M:%S - " + Agent.get_name())
        os.makedirs(self.folder_name)
        os.makedirs(self.folder_name + "/saved_networks")
        os.makedirs(self.folder_name + "/logs")

        #redirect stdout to log file instead
        self.old_stdout = sys.stdout
        self.log = open(self.folder_name+"/logs/init.log","a")
        sys.stdout = self.log

        # Generate a Torcs environment
        self.env = TorcsEnv(vision=self.vision, throttle=self.throttle, gear_change=self.gear_change)
        print("1. Env is created! with: vision="+str(self.vision) + ", throttle=" + str(self.throttle) +", gear_change=" + str(self.gear_change))

        # Create agent
        self.agent = Agent(env_name="TORCS", state_dim=self.state_dim, action_dim=self.action_dim)
        print("2. Agent is created! with state_dim=" + str(self.state_dim) + ", action_dim=" + str(self.action_dim))

        # create a settings file ( only for saving setting, not for applying settings!!!!
        self.settings_file = open(self.folder_name + "/" + "settings", "a")

        self.print_settings(settings_file=self.settings_file)  # print settings from runfile
        self.agent.print_settings(settings_file=self.settings_file)  # print settings from agent
        self.print_commits() # add vesrion / commit files to run folder!

        self.settings_file.close()

        self.result = results(folder=self.folder_name)


    def run_ddpg(self):
        ### for episode = 1, M
        for episode in range(self.episode_count):

            if(episode % self.log_size == 0):
                self.new_log(episode=episode)

            print("=============================================================")
            print(" starting episode: " + str(episode) +"/"+ str(self.episode_count))
            done = self.done
            total_reward = 0.

            # train_indicator is equal to is_training but set to false when testing every 20th episode!
            #train_indicator = (self.is_training and not((episode > 10) and (episode % 20 == 0)))
            train_indicator = (self.is_training and not ((not episode == 0) and (episode % self.test_frequency == 0)))
            #train_indicator = (self.is_training and not episode % self.test_frequency == 0) # testing at ep=0, sets base reward!!

            ### Initialize a random process N for action exploration
            #Done in ddpg_agent constructor... OU

            #TODO Early stop? - train indicator is not is_training, but wheter a test run is active or not!
            early_stop = self.do_early_stop(epsilon=self.epsilon, train_indicator=train_indicator)

            ### Receive initial observation state s_t
            # relaunch TORCS every 5 episode because of the memory leak error
            ob = self.env.reset(relaunch=((episode % 5) == 0))
            s_t = self.create_state(ob)


            ### for t = 1, T
            for step in range(self.max_steps):
                self.total_steps += 1

                # reduce epsilon
                if (train_indicator):
                    self.epsilon -= 1.0 / self.EXPLORE
                    self.epsilon = max(self.epsilon, 0.1)

                ### Execute action at and observe reward rt and observe new state st+1
                # 1. get that action (is_training=true gives noisy action!! / exploration)
                a_t = self.agent.act(s_t=s_t, is_training=train_indicator, epsilon=self.epsilon, done=done)

                # 2. send that action to the environment and observe rt and new state
                ob, r_t, done, info = self.env.step(a_t, early_stop)
                s_t1 = self.create_state(ob) # next state, after action a_t

                # Cheking for nan rewards
                if (math.isnan(r_t)):
                    r_t = 0.0
                    print('NAN reward <===========================================================================')


                ### Store transition (st,at,rt,st+1) in ReplayBuffer
                self.agent.replay_buffer.add(s_t, a_t, r_t, s_t1, done)


                total_reward += r_t
                s_t = s_t1

                ### training (includes 5 steps from ddpg algo):
                if(train_indicator and self.agent.replay_buffer.count() > self.agent.REPLAY_START_SIZE):
                    self.agent.train()

                # print info:
                if ((step % 20) == 0):
                    print("Ep:" + str(episode) + " step:" + str(step) + "(" + str(self.total_steps) + ")"
                          + ", a_t=[s={: .2f}, t={: f}, b={: f}]".format(a_t[0], a_t[1], a_t[2])
                          + ", r_t={: f}".format(r_t)
                          + ", Reward= {: .2f} / {: .2f}".format(total_reward, self.best_training_reward)
                          + ", epsilon= {: .3f}".format(self.epsilon)
                          + ", speed= {: .2f}".format(ob['speedX'] * 300)
                          + ", gear={: .0f}".format(ob['gear'])
                          + ", damage={: .2f}".format(ob['damage']))

                # so that this loop stops if torcs is restarting or done!
                if done:
                    break


            ### end for (en of episode)
            if(train_indicator):
                if (total_reward > self.best_training_reward):  # update best training reward
                    #print("Now we save model with reward " + str(total_reward) + " previous best reward was " + str(self.best_total_reward))
                    self.best_training_reward = total_reward
                    self.agent.save_networks(global_step=self.total_steps, run_folder=self.folder_name)
            else:
                if (total_reward > self.best_testing_reward): # update best testing reward
                    self.best_testing_reward = total_reward

            # add result to result saver! when testing #TODO remember to chang in result_instpecter if this is changed!
            self.result.add(row=[episode, self.total_steps, self.best_training_reward, self.best_testing_reward, total_reward, train_indicator, self.epsilon, early_stop])
            if(episode % 10 == 0) :
                self.result.save()
                # do some end of ep printing in regular terminal so its easier to see if running!
                self.log.flush()
                self.print_memory(episode=episode)




        ### end for end of all episodes!
        self.finish()


    def print_settings(self, settings_file):
        settings_text = ["==== from runfile ====" + "\n",
                        "is_training = " + str(self.is_training) + "\n",
                         "epsilon_start = " + str(self.epsilon_start) + "\n",
                         "episode_count = " + str(self.episode_count) + "\n",
                         "max_steps = " + str(self.max_steps) + "\n",
                         "vision = " + str(self.vision) + "\n",
                         "throttle = " + str(self.throttle) + "\n",
                         "gear_change = " + str(self.gear_change) + "\n",
                         "state_dim = " + str(self.state_dim) + "\n",
                         "action_dim = " + str(self.action_dim)]
        for line in settings_text:
            settings_file.write(line)  # print settings to file

    def print_commits(self):
        # get version / commit of current folder/gym_torcs
        os.system("echo $(git log -n 1) > ./gym_torcs_version")
        os.system("cp ./gym_torcs_version " + self.folder_name.replace(" ", "\ ") + "/gym_torcs_version")
        os.system("cp ./agent_version " + self.folder_name.replace(" ", "\ ") + "/agent_version")

    def print_memory(self, episode):
        file = self.folder_name.replace(" ", "\ ") + "/logs/memory.log"
        os.system("echo 'Episode "+ str(episode) + "' >> " + file )
        os.system("echo $(free -m) >> " + file)



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
        # 1. original sensors!!!
        #s_t = np.hstack((ob['angle'], ob['track'], ob['trackPos'], ob['speedX'], ob['speedY'], ob['speedZ'], ob['wheelSpinVel'], ob['rpm'], ob['gear']/6))

        # 2. realistic sensors!! 5+ 36+ 19+ 1+ 1+ 1+ 4+ 1
        #s_t = np.hstack((ob['focus'], ob['opponents'], ob['track'], ob['speedX'], ob['speedY'], ob['speedZ'], ob['wheelSpinVel'], ob['rpm'], ob['gear']/6))

        # 3. combo! for driving without vision, but close to realistic!
        s_t = np.hstack((ob['angle'],ob['track'], ob['trackPos'], ob['focus'], ob['opponents'], ob['track'], ob['speedX'], ob['speedY'], ob['speedZ'], ob['wheelSpinVel'], ob['rpm'], ob['gear']/6))
        #print("s_t.shape=" + str(s_t.shape))
        return s_t

    def do_early_stop(self, epsilon, train_indicator):
        random_number = random.random()
        eps_early = max(epsilon, 0.10)
        return (random_number < (1.0 - eps_early)) and (train_indicator == 1)

    # everything that should be done at end of run!
    def finish(self):
        # add finished to the run folder!
        os.system("mv " + self.folder_name.replace(" ", "\ ")  + " " + (self.folder_name+" FINISHED").replace(" ", "\ "))
        self.env.end()  # This is for shutting down TORCS
        print("Run finnished at : " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        sys.stdout = self.old_stdout

    def new_log(self, episode):
        self.log = open(self.folder_name + "/logs/run-"+str(episode)+".log", "a")
        sys.stdout = self.log


if __name__ == "__main__":
    runner = agent_runner()
    runner.run_ddpg()


