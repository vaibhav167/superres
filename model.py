import torch.nn as nn
import torch
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv3 = nn.Conv2d(3, 3, 3, 1, 1)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(3, 3, 3, 1, 1)
    
    def forward(self, x):
        x = self.upsample1(F.relu(self.conv1(x)))
        x = self.upsample2(F.relu(self.conv2(x)))
        x = self.upsample3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        return x