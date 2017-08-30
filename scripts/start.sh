#!/bin/bash
echo "Launch TORCS simulation"

# get docker image
echo "Pull the docker image..."
sudo docker pull kloek/torcsproject
echo "...done"
echo ""

# TODO select config file before running?

# start the virtual gui
echo "Run the docker image..."
id="$(sudo docker run -td kloek/torcsproject Xvfb :1 -screen 0 800x600x16)"
echo "...done, running with id = $id"
echo ""

# update the repo
echo "Update the gym_torcs and agents repo..."
sudo docker exec -t $id git pull origin master
sudo docker exec -t $id sh -c 'cd /TorcsProject; git pull origin master'
echo "...done"
echo ""

echo "move agents to gym_torcs"
sudo docker exec -t $id sh -c 'cd /TorcsProject; ./install.sh'
echo "... done"
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

echo "Move results out from docker"
sudo docker cp $id:~/Anton/runs/* ~/Anton/runs
echo "...done"


#echo "Stop and remove dockers"
#sudo docker stop $id
#sudo docker rm $id