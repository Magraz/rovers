from collections import deque
import numpy as np
from random import sample
import torch
from torch.utils.data import Dataset, DataLoader
from pyrover_domain.fitness_critic.models.mlp import MLP_Model


class TrajectoryRewardDataset(Dataset):

    def __init__(self, traj_hist):

        if len(traj_hist) < 256:
            trajG = traj_hist
        else:
            trajG = sample(traj_hist, 256)

        self.observations, self.reward = [], []

        for traj, g in trajG:
            for s in traj:  # train whole trajectory
                self.observations.append(s)
                self.reward.append([g])

        self.observations, self.reward = np.array(self.observations), np.array(self.reward)

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):

        return (
            self.observations[idx],
            self.reward[idx],
        )


class FitnessCritic:
    def __init__(self, device: str, model_type: str, loss_fn: int, episode_size: int):

        self.episode_size = episode_size
        self.hist = deque(maxlen=30000)
        self.device = device

        self.model_type = model_type

        match self.model_type:
            case "MLP":
                self.model = MLP_Model(loss_fn=loss_fn).to(device)

    def add(self, trajectory, G):
        self.hist.append((trajectory, G))

    def evaluate(self, trajectory):  # evaluate max state
        result = self.model.forward(torch.from_numpy(trajectory).to(self.device)).cpu().detach().numpy()
        return np.max(result)

    def train(self, epochs: int = 3):
        traj_dataset = None
        loss_arr = []

        match self.model_type:
            case "MLP":
                traj_dataset = TrajectoryRewardDataset(self.hist)

        for _ in range(epochs):

            dataloader = DataLoader(traj_dataset, batch_size=self.episode_size, shuffle=True, num_workers=0)

            for x, y in dataloader:
                loss_arr.append(self.model.train(x.to(self.device), y.to(self.device)))
