from collections import deque
import numpy as np
from random import sample
import torch
from pyrover_domain.fitness_critic.models.mlp import MLP_Model


class FitnessCritic:
    def __init__(self, device: str, model_type: str, loss_f: int):

        match model_type:
            case "MLP":
                self.model = MLP_Model(loss_fn=loss_f).to(device)

        self.hist = deque(maxlen=30000)

        self.device = device

    def add(self, trajectory, G):
        self.hist.append([trajectory, G])

    def evaluate(self, trajectory):  # evaluate max state
        result = self.model.forward(torch.from_numpy(trajectory).to(self.device)).cpu().detach().numpy()
        return np.max(result)

    def train(self):
        for _ in range(20):

            if len(self.hist) < 256:
                trajG = self.hist
            else:
                trajG = sample(self.hist, 256)

            S, G = [], []

            for traj, g in trajG:
                for s in traj:  # train whole trajectory
                    S.append(s)
                    G.append([g])

            S, G = np.array(S), np.array(G)

            self.model.train(
                torch.from_numpy(S).to(self.device),
                torch.from_numpy(G).to(self.device),
            )
