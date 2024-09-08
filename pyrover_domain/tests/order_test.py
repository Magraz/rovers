from pyrover_domain.librovers import rovers
from tqdm import tqdm
from pyrover_domain.custom_env import createEnv

import os
from pathlib import Path
import yaml

import numpy as np

import logging

# Create and configure logger
logging.basicConfig(format="%(asctime)s %(message)s")

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)

print("Setting up bindings...")

config_dir = Path(os.path.expanduser("/home/magraz/rovers/pyrover_domain/experiments/yamls/ordered_mlp.yaml"))

with open(str(config_dir), "r") as file:
    config = yaml.safe_load(file)

# 4 sectors times 2 distance types from pois and rovers
state_len = 8

env = createEnv(config)

theta = np.linspace(-0.75 * np.pi, 0.25 * np.pi, config["ccea"]["num_steps"])

states, rewards = env.reset()
for s in range(config["ccea"]["num_steps"]):

    for i, rover in enumerate(env.rovers()):
        print(f"ROVER: {i},  POS: {rover.position()}")

    for i, poi in enumerate(env.pois()):
        pack = rovers.EntityPack(entity=poi, agents=env.rovers(), entities=env.pois())

        print(f"{s} POI: {i}, ORDER: {poi.order}, SATISFIED: {poi.constraint_satisfied(pack)} VAL: {poi.value()}\n")

    radius = 21

    h = 25
    k = 25

    dx = -(rover.position().x - (radius * np.cos(theta[s]) + h))
    dy = -(rover.position().y - (radius * np.sin(theta[s]) + k))

    norm = np.sqrt(dx**2 + dy**2)

    norm_dx = dx / norm
    norm_dy = dy / norm

    if s == 0:
        action = [rovers.tensor([dx, dy]), rovers.tensor([dx, dy])]
    else:
        action = [rovers.tensor([norm_dx, norm_dy]), rovers.tensor([norm_dx, norm_dy])]
        # dummy_action = [rovers.tensor([0,0])]

    for state in states:
        obs_tensor = state.data()
        obs_tensor.reshape((8,))  # State space is 8 dimensional
        obs_tensor = np.frombuffer(obs_tensor, dtype=np.float64, count=8)

    states, rewards = env.step(action)
