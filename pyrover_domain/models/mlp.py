import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class MLP_Policy(nn.Module):  # inheriting from nn.Module!

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLP_Policy, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.double)
        self.output = nn.Linear(hidden_size, output_size, dtype=torch.double)
        self.num_params = nn.utils.parameters_to_vector(self.parameters()).size()[0]

    def forward(self, x: torch.Tensor):
        out = F.leaky_relu(self.fc1(x))
        out = self.output(out)
        return F.tanh(out)
    
    def get_params(self):
        return nn.utils.parameters_to_vector(self.parameters())

    def set_params(self, params: torch.Tensor):
        nn.utils.vector_to_parameters(params, self.parameters())
    
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MLP_Policy(input_size=8, hidden_size=32, output_size=2).to(device)
    model_copy = deepcopy(model)
    print(model_copy.num_params)

    input = torch.tensor([-1,-1,-1,-1,-1,-1,-1,-1], dtype=torch.double).to(device)
    print(model_copy.forward(input))

    rand_params = torch.rand(model_copy.get_params().size()).to(device)
    mutated_params = torch.add(model_copy.get_params(), rand_params).to(device)

    model_copy.set_params(mutated_params)

    print(model_copy.forward(input))


