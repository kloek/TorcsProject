#!/bin/bash
Xvfb :1 -screen 0 800x600x16 &
x11vnc -forever -create -display :1.0 &
export DISPLAY=:1.0