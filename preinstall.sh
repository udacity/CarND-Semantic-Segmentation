#!/bin/bash
# May need to uncomment and update to find current packages
# apt-get update
wget https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/lyft_training_data.tar.gz
tar -xzf lyft_training_data.tar.gz

# Required for demo script! #
pip install scikit-video

# Add your desired packages for each workspace initialization
#          Add here!          #
