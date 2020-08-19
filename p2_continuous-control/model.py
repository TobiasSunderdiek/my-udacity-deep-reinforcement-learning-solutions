import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc_1 = nn.Linear(input_size, 400)
        self.fc_2 = nn.Linear(400, 300)
        self.fc_3 = nn.Linear(300, output_size)

    def forward(self, input):
        x = F.leaky_relu(self.fc_1(input))
        x = F.leaky_relu(self.fc_2(x))
        x = self.fc_3(x) #todo add leaky relu?

        return x

