#!/usr/bin/env bash


#r1: sett reward function to as similar to original as possible: (gym_torcs r1)
#git checkout r1

#a without gears @ port=3101
#rm config.py
#cp multiple_runns/configs/config_r1a.py config.py
#python3 run_ddpg.py

#b with gears @ port=3102
#rm config.py
#cp multiple_runns/configs/config_r1b.py config.py
#python3 run_ddpg.py

##########################################################################
#r2: sett reward function to same as r1 but with new damage! (gym_torcs r2)
#git checkout r2

#a without gears
#rm config.py
#cp multiple_runns/configs/config_r2a.py config.py
#python3 run_ddpg.py

#b with gears
#rm config.py
#cp multiple_runns/configs/config_r2b.py config.py
#python3 run_ddpg.py

###########################################################################
#r3: set reward function to same as r2 but with x3 progress! (gym_torcs r3)
#git checkout r3

#a without gears
#rm config.py
#cp multiple_runns/configs/config_r3a.py config.py
#python3 run_ddpg.py

#b with gears
#rm config.py
#cp multiple_runns/configs/config_r3b.py config.py
#python3 run_ddpg.py


###########################################################################
#r4: Latest reward function (gym_torcs master)
#git checkout master

#a without gears
#rm config.py
#cp multiple_runns/configs/config_r4a.py config.py
#python3 run_ddpg.py

#b with gears
#rm config.py
#cp multiple_runns/configs/config_r4b.py config.py
#python3 run_ddpg.py

#AM : MEMORY LOGGING????
#rm config.py
#cp multiple_runns/configs/config_r4am.py config.py
#python3 run_ddpg.py

#r5: Latest reward function + Safety Critic v1
#a without gears
#rm config.py
#cp multiple_runns/configs/config_r5a.py config.py
#python3 run_ddpg.py

#b with gears
#rm config.py
#cp multiple_runns/configs/config_r5b.py config.py
#python3 run_ddpg.py

#r6: Latest reward function + Safety Critic v2
#a without gears
#rm config.py
#cp multiple_runns/configs/config_r6a.py config.py
#python3 run_ddpg.py

#b with gears
#rm config.py
#cp multiple_runns/configs/config_r6b.py config.py
#python3 run_ddpg.py

#r7a: add safety critic gamma and test a couple: WITHOUT GEARS
#r71: safety_gamma = 0.99
#rm config.py
#cp multiple_runns/configs/r7a/config_r71.py config.py
#python3 run_ddpg.py

#r72: safety_gamma = 0.97
#rm config.py
#cp multiple_runns/configs/r7a/config_r72.py config.py
#python3 run_ddpg.py

#r73: safety_gamma = 0.95
#rm config.py
#cp multiple_runns/configs/r7a/config_r73.py config.py
#python3 run_ddpg.py

#r74: safety_gamma = 0.93
#rm config.py
#cp multiple_runns/configs/r7a/config_r74.py config.py
#python3 run_ddpg.py

#r75: safety_gamma = 0.91
#rm config.py
#cp multiple_runns/configs/r7a/config_r75.py config.py
#python3 run_ddpg.py

#==========================================================

#r7b: add safety critic gamma and test a couple: WITH GEARS
#r71: safety_gamma = 0.99
rm config.py
cp multiple_runns/configs/r7b/config_r71.py config.py
python3 run_ddpg.py

#r72: safety_gamma = 0.97
rm config.py
cp multiple_runns/configs/r7b/config_r72.py config.py
python3 run_ddpg.py

#r73: safety_gamma = 0.95
rm config.py
cp multiple_runns/configs/r7b/config_r73.py config.py
python3 run_ddpg.py

#r74: safety_gamma = 0.93
rm config.py
cp multiple_runns/configs/r7b/config_r74.py config.py
python3 run_ddpg.py

#r75: safety_gamma = 0.91
rm config.py
cp multiple_runns/configs/r7b/config_r75.py config.py
python3 run_ddpg.py


