import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_Policy(nn.Module):  # inheriting from nn.Module!

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(MLP_Policy, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.double)
        self.output = nn.Linear(hidden_size, output_size, dtype=torch.double)
        self.num_params = nn.utils.parameters_to_vector(self.parameters()).size()[0]

    def forward(self, x: torch.Tensor):
        out = F.relu(self.fc1(x))
        out = self.output(out)
        return F.tanh(out)
    
    def get_params(self):
        return nn.utils.parameters_to_vector(self.parameters())

    def set_params(self, params: torch.Tensor):
        nn.utils.vector_to_parameters(params, self.parameters())
    
if __name__ == "__main__":
    model = MLP_Policy(input_size=8, hidden_size=32, output_size=2)
    print(model.num_params)

    input = torch.tensor([-1,-1,-1,-1,-1,-1,-1,-1], dtype=torch.double)
    print(model.forward(input))

    rand_params = torch.rand(model.get_params().size())
    mutated_params = torch.add(model.get_params(), rand_params)

    model.set_params(mutated_params)

    print(model.forward(input))


