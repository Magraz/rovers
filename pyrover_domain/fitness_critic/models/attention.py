import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, device: str
) -> torch.Tensor:

    # Efficient implementation equivalent to the following:
    L, D = query.size(-2), key.size(-1)
    scale_factor = 1 / math.sqrt(D)
    attn_bias = torch.zeros(L, L, dtype=query.dtype).to(device)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return attn_weight @ value, attn_weight


def getPositionEncoding(seq_len: int, d: int, n: int = 10000):
    P = np.zeros((seq_len, d))

    for k in range(seq_len):

        for i in np.arange(int(d / 2)):

            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)

    return P


class Attention_Model(nn.Module):  # inheriting from nn.Module!

    def __init__(
        self,
        device: str,
        loss_fn: int,
        seq_len: int,
        in_dim: int = 8,
        hid_dim: int = 50,
        lr: float = 1e-3,
    ):
        super(Attention_Model, self).__init__()

        self.device = device

        self.w1 = nn.Linear(in_dim, hid_dim)

        self.w_out1 = nn.Linear(hid_dim, hid_dim)
        self.w_out2 = nn.Linear(hid_dim, 1)
        self.w_out3 = nn.Linear(seq_len, 1)

        self.pos_enc = getPositionEncoding(seq_len, in_dim)
        self.pos_enc = torch.from_numpy(self.pos_enc.astype(np.float64)).to(self.device)

        self.to(torch.double)

        # Set loss function
        if loss_fn == 0:
            self.loss_func = nn.MSELoss(reduction="sum")
        elif loss_fn == 1:
            self.loss_func = self.alignment_loss
        elif loss_fn == 2:
            self.loss_func = lambda x, y: self.alignment_loss(x, y) + nn.MSELoss(reduction="sum")(x, y)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor):

        KQV = self.w1(x + self.pos_enc)
        res, attn = scaled_dot_product_attention(KQV, KQV, KQV, self.device)
        transformout = self.w_out2(F.leaky_relu(self.w_out1(res)))
        transformout = torch.flatten(transformout, start_dim=-2, end_dim=-1)

        return self.w_out3(transformout)

    def train(self, x: torch.Tensor, y: torch.Tensor, shaping=False):

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
