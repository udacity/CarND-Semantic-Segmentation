#!/bin/bash
# connect to the instance with security key in ~/.ssh/

# Install dependencies
echo -e "\n installing dependencies"
pip install tensorflow-gpu
pip install tqdm

# cloning the Semantic Segmentation Repo
echo -e "Cloning Semantic Segmentatio"
git clone https://github.com/robroooh/CarND-Semantic-Segmentation.git
cd CarND-Semantic-Segmentation
python main.py