#!/bin/bash

gym_torcs="../gym_torcs"
TorcsProjectVersion="TorcsProjectVersion"

# version / commit installed
git log -n 1 > $gym_torcs/agent_version
git status >> $gym_torcs/agent_version
git diff >> $gym_torcs/agent_version

# agents folder
rm -rf $gym_torcs/agents
cp -r agents $gym_torcs/agents

# run file
rm $gym_torcs/run_ddpg.py
cp run_ddpg.py $gym_torcs/

# config
rm $gym_torcs/config.py
cp config.py $gym_torcs/

# multirun folder
rm -rf $gym_torcs/multiple_runs
cp -r multiple_runns $gym_torcs/
cp $gym_torcs/multiple_runns/multiple_runns.sh $gym_torcs/multiple_runns.sh
chmod +x $gym_torcs/multiple_runns.sh

# xml files for torcs
cp docker/practice.xml ~/.torcs/config/raceman
cp docker/practice.xml /usr/local/share/games/torcs/config/raceman
cp docker/screen.xml  ~/.torcs/config
cp docker/screen.xml  /usr/local/share/games/torcs/config
