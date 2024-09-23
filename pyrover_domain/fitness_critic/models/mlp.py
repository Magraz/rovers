import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class MLP_Model(nn.Module):  # inheriting from nn.Module!

    def __init__(
        self,
        loss_fn: int,
        input_size: int = 8,
        hidden_layers: int = 2,
        hidden_size: int = 80,
        lr: float = 1e-3,
    ):
        super(MLP_Model, self).__init__()

        self.hidden_layers = hidden_layers

        match (self.hidden_layers):
            case 1:
                self.fc1 = nn.Linear(input_size, hidden_size)
            case 2:
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.output = nn.Linear(hidden_size, 1)

        self.to(torch.double)

        # Set loss function
        if loss_fn == 0:
            self.loss_func = nn.MSELoss(reduction="sum")
        elif loss_fn == 1:
            self.loss_func = self.alignment_loss
        elif loss_fn == 2:
            self.loss_func = lambda x, y: self.alignment_loss(x, y) + nn.MSELoss(reduction="sum")(x, y)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        out = F.tanh(self.fc1(x))

        match (self.hidden_layers):
            case 2:
                out = F.tanh(self.fc2(out))

        return self.output(out)

    def train(self, x, y, shaping=False):

        self.optimizer.zero_grad()
        pred = self.forward(x)
        loss = self.loss_func(pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.cpu().detach().item()

    def alignment_loss(self, o, t):
        ot = torch.transpose(o, 0, 1)
        tt = torch.transpose(t, 0, 1)

        O = o - ot
        T = t - tt

        align = torch.mul(O, T)
        align = F.sigmoid(align)
        loss = -torch.mean(align)

        return loss
