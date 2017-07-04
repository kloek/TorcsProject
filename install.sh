#!/bin/bash

folder="$HOME/Documents/Thesis_work/gym_torcs/"

# agents folder
rm -rf $folder/agents
cp -r agents $folder/agents

# run file
rm $folder/run_ddpg.py
cp run_ddpg.py $folder