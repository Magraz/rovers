import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pyrover_domain.utils.loss_functions import alignment_loss


class MLP_Model(nn.Module):  # inheriting from nn.Module!

    def __init__(
        self,
        loss_func,
        input_size: int = 8,
        hidden_layers: int = 2,
        hidden_size: int = 80,
        lr: float = 1e-3,
    ):
        super(MLP_Model, self).__init__()

        self.hidden_layers = hidden_layers

        self.loss_func = loss_func

        match (self.hidden_layers):
            case 1:
                self.fc1 = nn.Linear(input_size, hidden_size)
            case 2:
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.output = nn.Linear(hidden_size, 1)

        self.double()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def get_params(self):
        return nn.utils.parameters_to_vector(self.parameters())

    def forward(self, x: torch.Tensor):
        out = F.tanh(self.fc1(x))

        match (self.hidden_layers):
            case 2:
                out = F.tanh(self.fc2(out))

        return self.output(out)

    def train(self, x: torch.Tensor, y: torch.Tensor):

        self.optimizer.zero_grad()
        pred = self.forward(x)
        loss = self.loss_func(pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.cpu().detach().item()
