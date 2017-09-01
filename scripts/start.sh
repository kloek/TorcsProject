#!/bin/bash
echo "Launch TORCS simulation"

# get docker image
echo "Pull the docker image..."
sudo docker pull kloek/torcsproject
echo "...done"
echo ""

# start the virtual gui
echo "Run the docker image..."
id="$(sudo docker run -td kloek/torcsproject Xvfb :1 -screen 0 800x600x16)"
echo "...done"
echo ""

# update the repo
echo "Update the gym_torcs repo..."
sudo docker exec -t $id git pull origin master
echo "...done"
echo ""

# update the repo
echo "Update the TorcsProject repo..."
sudo docker exec -t $id sh -c 'cd /TorcsProject; git pull origin master; ./install.sh'
echo "...done"
echo ""

# create the fake display to run torcs
echo "Open fake display..."
sudo docker exec -td $id x11vnc -forever -create -display :1.0
echo "...done"
echo ""

# start the simultation
echo "Starting the simulation !"
echo ""
#sudo docker exec -t -e "DISPLAY=:1.0" $(sudo docker ps -lq) bash -c "python run_ddpg.py"
sudo docker exec -t -e "DISPLAY=:1.0" $id bash -c "python3 run_ddpg.py"
