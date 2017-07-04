# -*- coding: utf-8 -*-

from gym_torcs import TorcsEnv
from agents.ddpg.ddpg_agent import Agent
import numpy as np
import random

# Notes for readability: comments with tripple sharp (###) is the steps of main algorithm
# these tripple comments are spread across multiple files since diffrent part of algo is performed by diffrent classes
# such as run_ddpg, which contains eppisode and steps loops... ddpg_agent which contains agent related functions, and so on....
# ddpg from : https://arxiv.org/pdf/1509.02971.pdf

is_training =  1  #TODO sys arg or config file
epsilon_start  = 1.0 #TODO sys arg or config file

def run_ddpg():
    # Run time
    episode_count = 10 #TODO sys arg or config file
    max_steps = 50 #TODO sys arg or config file

    # initial values
    reward = 0
    done = False
    epsilon = epsilon_start


    # Gym_torcs
    vision = False
    throttle = True
    gear_change = True
    #brake = true #TODO

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=throttle, gear_change=gear_change)
    print("1. Env is created!")

    #TODO ALL THIS IS EXPERIMENTAL!!!!!
    #TODO, why is action state size two when it should be 3?
    #TODO, use this to set action_dim and state_dim
    #action_dim = env.action_space.sample().shape
    #print("action_dim = " + str(action_dim))
    #state_dim = env.observation_space.sample()
    #s_t = env.observation_space.sample()
    #print("State size" + str(s_t.shape))
    #print("sample does work")
    #print("action_dim=" + str(action_dim) + ",  state_dim=" + str(state_dim))



    # initiate an agent, construct includes some of ddpg algo steps
    state_dim = 29 #TODO
    action_dim = 3
    agent = Agent(state_dim=state_dim, action_dim=action_dim)
    print("2. Agent is created!")

    #TODO implement printing all settings here!
    print("==========================================================")
    print("TORCS Experiment Starting with: " + str(agent.get_name))
    print("vision= " + str(vision) + ", throttle= " + str(throttle) + ", gear_change= " + str(gear_change))
    #TODO print state and action dim
    print("episode_count= " + str(episode_count) + ", max_steps= " + str(max_steps))



    ### for episode = 1, M
    for episode in range(episode_count):

        # train_indicator is equal to is_training but set to false when testing
        train_indicator = (is_training and not((episode > 10) and (episode % 20 == 0)))

        ### Initialize a random process N for action exploration
        #TODO Done in ddpg_agent constructor...

        #TODO Early stop? - train indicator is not is_training, but wheter a test run is active or not!
        early_stop = do_early_stop(epsilon, train_indicator)

        ### Receive initial observation state s_t
        # relaunch TORCS every 3 episode because of the memory leak error
        ob = env.reset(relaunch=((episode % 3) == 0))
        s_t = create_state(ob)


        ### for t = 1, T
        for step in range(max_steps):

            ### Execute action at and observe reward rt and observe new state st+1
            a_t = agent.act(s_t=s_t, is_training=is_training, done=done)

            #TODO TEST-CODE to be able to run before act is implemented!
            #if a_t == []:
               # a_t = env.action_space.sample()
               # print("action="+str(a_t))

            # send that action to the environment
            ob, r_t, done, info = env.step(a_t)
            s_t1 = create_state(ob) # next state, after action a_t

            ### Store transition (st,at,rt,st+1) in ReplayBuffer
            agent.replay_buffer.add(s_t, a_t, r_t, s_t1, done)


            ### training (includes 5 steps from ddpg algo):
            if(train_indicator):
                agent.train()

        ### end for
    ### end for

def create_state(ob):
    # TODO this is without vision!!!!!
    # print("observation=" + str(ob)) #TODO; ob doesnt contain the right stuff! Make sure it contains everything!!!
    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel / 100.0, ob.rpm))
    return s_t

def do_early_stop(epsilon, train_indicator):
    random_number = random.random()
    eps_early = max(epsilon, 0.10)
    return (random_number < (1.0 - eps_early)) and (train_indicator == 1)

if __name__ == "__main__":
    run_ddpg()
