import torch.nn as nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x=x.view(-1, 28*28)
        outputs = self.linear(x)
        return outputs

