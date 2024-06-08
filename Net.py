import torch
import torch.nn as nn
import torch.nn.functional as F
from dlgo import zero

class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()

        self.layers =  nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=11, out_channels=11, kernel_size=3, padding=1),
                nn.BatchNorm2d(11),
                nn.ReLU()
            ) for _ in range(4)
        ])

        # Policy output
        self.policy_conv = nn.Conv2d(in_channels=11, out_channels=2, kernel_size=1)
        self.policy_batch = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU()
        self.policy_flat = nn.Flatten()
        self.policy_output = nn.Linear(2 * 9 * 9, encoder.num_moves()) #board_size
        self.policy_softmax = nn.Softmax(dim=1)

        # Value output
        self.value_conv = nn.Conv2d(in_channels=11, out_channels=1, kernel_size=1)
        self.value_batch = nn.BatchNorm2d(1)
        self.value_relu = nn.ReLU()
        self.value_flat = nn.Flatten()
        self.value_hidden = nn.Linear(1 * 9 * 9, 256) #board_size
        self.value_relu_hidden = nn.ReLU()
        self.value_output = nn.Linear(256, 1)
        self.value_tanh = nn.Tanh()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        # Policy output
        policy_x = self.policy_conv(x)
        policy_x = self.policy_batch(policy_x)
        policy_x = self.policy_relu(policy_x)
        policy_x = self.policy_flat(policy_x)
        policy_x = self.policy_output(policy_x)
        policy_x = self.policy_softmax(policy_x)

        # Value output
        value_x = self.value_conv(x)
        value_x = self.value_batch(value_x)
        value_x = self.value_relu(value_x)
        value_x = self.value_flat(value_x)
        value_x = self.value_hidden(value_x)
        value_x = self.value_relu_hidden(value_x)
        value_x = self.value_output(value_x)
        value_x = self.value_tanh(value_x)

        return policy_x, value_x
    
