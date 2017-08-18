#!/bin/bash
for pid in `sudo docker ps -q`; do
	sudo docker exec $pid pkill -9 python
	sudo docker cp $pid:rl_monitor/results ~
	sudo docker cp $pid:rl_monitor/models ~
done
sudo docker stop $(sudo docker ps -q)
sudo docker rm $(sudo docker ps -a -q)
