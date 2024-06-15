from pyrover_domain.librovers import rovers, thyme
import numpy as np
from tqdm import tqdm
from custom_pois import DecayPOI
from custom_sensors import CustomLidar
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

# logging.basicConfig(level=logging.DEBUG)


"""
================================================================================
Putting it all together.
This file uses all the defined variations of \
    1. Agents
    2. Sensors
    3. Entities
    4. Rewards
to create a rich customizable and heterogeneous environment.

A PyTorch stub near the end of the file shows an example of running an episode.
================================================================================
"""

"""
================================================================================
Setting up the environment
================================================================================
"""
print ("Setting up bindings...")

# aliasing some types to reduce typing
Discrete = thyme.spaces.Discrete            # Discrete action space
Reward = rovers.rewards.Difference      # Difference reward

num_agents = 4
obs_radius = 3
resolution = 90

agents = [
    rovers.Rover[CustomLidar, Discrete, Reward](obs_radius, CustomLidar(resolution=resolution, composition_policy=rovers.Density()), Reward()) for _ in range(num_agents)
]

num_pois = 3
value = 10
obs_radius = 2.0
coupling = 3
decay_rate = 1

# pois = [
#     DecayPOI(value, obs_radius, rovers.CountConstraint(coupling), decay_rate) for _ in range(num_pois)
# ]

pois = [
    DecayPOI(value, obs_radius, rovers.CountConstraint(coupling), decay_rate) for _ in range(num_pois)
]

# 4 sectors times 2 distance types from pois and rovers
state_len = 8

Env = rovers.Environment[rovers.CornersInit]
env = Env(rovers.CornersInit(10.0), agents, pois)
states, rewards = env.reset()

print ("Sample environment state (state of each rover is a row): ")
for state in states:
    s = state.data() 
    s.reshape((state_len,))
    s = np.frombuffer(s, dtype=np.float32, count=state_len)
    print(s)

"""
================================================================================
Setting up a policy
================================================================================
"""
class MLP_Policy(nn.Module):  # inheriting from nn.Module!

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_Policy, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.output(out)
        return out

"""
================================================================================
Running n steps
================================================================================
"""

print ("\nBegining learning sequence.")

policies = [MLP_Policy(input_size=8, hidden_size=32, output_size=2) for _ in agents]

states, _ = env.reset()
steps = 1000

for s in tqdm(range(steps)):

    actions = []

    for i in range(len(agents)):

        states_tensor = states[i].data() 
        states_tensor.reshape((state_len,))
        states_tensor = np.frombuffer(states_tensor, dtype=np.float32, count=state_len)
        states_tensor = torch.from_numpy(states_tensor)

        action = policies[i].forward(states_tensor)

        actions.append(rovers.tensor(action.detach().numpy()))
    
    states, rewards = env.step(actions)
    
    print("\nStates:")
    for ind, state in enumerate(states):
        print("agent "+str(ind))
        print(state.transpose())

    print("\nRewards:")
    for ind, reward in enumerate(rewards):
        print("reward "+str(ind))
        print(reward)

    # for i in range(len(agents)):
    #     learn(old_states[i], actions[i], states[i], rewards[i])

print ("Learning complete.")
