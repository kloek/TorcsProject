#!/usr/bin/env bash


#r1: sett reward function to as similar to original as possible: (gym_torcs r1)
git checkout r1

#a without gears @ port=3101
rm config.py
cp multiple_runns/configs/config_r1a.py config.py
python3 run_ddpg.py

#b with gears @ port=3102
rm config.py
cp multiple_runns/configs/config_r1b.py config.py
python3 run_ddpg.py

##########################################################################
#r2: sett reward function to same as r1 but with new damage! (gym_torcs r2)
git checkout r2

#a without gears
rm config.py
cp multiple_runns/configs/config_r2a.py config.py
python3 run_ddpg.py

#b with gears
rm config.py
cp multiple_runns/configs/config_r2b.py config.py
python3 run_ddpg.py

###########################################################################
#r3: set reward function to same as r2 but with x3 progress! (gym_torcs r3)
git checkout r3

#a without gears
rm config.py
cp multiple_runns/configs/config_r3a.py config.py
python3 run_ddpg.py

#b with gears
rm config.py
cp multiple_runns/configs/config_r3b.py config.py
python3 run_ddpg.py


###########################################################################
#r4: Latest reward function (gym_torcs master)
git checkout master

#a without gears
rm config.py
cp multiple_runns/configs/config_r4a.py config.py
python3 run_ddpg.py

#b with gears
rm config.py
cp multiple_runns/configs/config_r4b.py config.py
python3 run_ddpg.py

#AM : MEMORY LOGGING????
rm config.py
cp multiple_runns/configs/config_r4am.py config.py
python3 run_ddpg.py

#r5: Latest reward function + Safety Critic v1
#a without gears
#b with gears

#r6: Latest reward function + Safety Critic v2
#a without gears
#b with gears

#r7: add safety critic gamma and test a couple:
#r71: safety_gamma = 0.99
#r72: safety_gamma = 0.97
#r73: safety_gamma = 0.95
#r74: safety_gamma = 0.93
#r75: safety_gamma = 0.91
