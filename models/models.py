from .BasicModule import BasicModule
from .Capsule_Pytorch import Capsule

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(BasicModule):
    def __init__(self, opt):
        super(CNN, self).__init__()
        self.opt = opt
        self.model_name = 'CNN'

        self.Conv2d1 = nn.Conv2d(1, 64, (3, 3))
        self.Conv2d2 = nn.Conv2d(64, 64, (3, 3))

        self.Conv2d3 = nn.Conv2d(64, 128, (3, 3))
        self.Conv2d4 = nn.Conv2d(128, 128, (3, 3))

        self.Dense1 = nn.Linear(128, 128)
        self.Dense2 = nn.Linear(128, 10)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.Conv2d1.weight)
        nn.init.xavier_normal_(self.Conv2d2.weight)
        nn.init.xavier_normal_(self.Conv2d3.weight)
        nn.init.xavier_normal_(self.Conv2d4.weight)
        nn.init.xavier_normal_(self.Dense1.weight)
        nn.init.xavier_normal_(self.Dense2.weight)

    def forward(self, x, y=None):

        x = self.Conv2d1(x).relu()
        x = self.Conv2d2(x).relu()
        x = F.avg_pool2d(x, (2, 2))

        x = self.Conv2d3(x).relu()
        x = self.Conv2d4(x).relu()

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.squeeze()

        x = self.Dense1(x).relu()
        out = self.Dense2(x).sigmoid()

        if y is None:
            return out
        loss = y * torch.relu(0.9 - out)**2 + 0.25*(1 - y)*torch.relu(out - 0.1)**2

        return torch.sum(loss)


class CNN_Capsule(BasicModule):
    def __init__(self, opt):
        super(CNN_Capsule, self).__init__()
        self.opt = opt
        self.model_name = 'CNN_Capusule'

        self.Conv2d1 = nn.Conv2d(1, 64, (3, 3))
        self.Conv2d2 = nn.Conv2d(64, 64, (3, 3))

        self.Conv2d3 = nn.Conv2d(64, 128, (3, 3))
        self.Conv2d4 = nn.Conv2d(128, 128, (3, 3))

        self.capsule = Capsule(input_dim=128, num_capsule=10, dim_capsule=16, routings=3, share_weights=True)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.Conv2d1.weight)
        nn.init.xavier_normal_(self.Conv2d2.weight)
        nn.init.xavier_normal_(self.Conv2d3.weight)
        nn.init.xavier_normal_(self.Conv2d4.weight)

    def forward(self, x, y=None):

        x = self.Conv2d1(x).relu()
        x = self.Conv2d2(x).relu()
        x = F.avg_pool2d(x, (2, 2))
        x = self.Conv2d3(x).relu()
        x = self.Conv2d4(x).relu()               # (B, 128, W, H)

        x = x.view(x.size(0), x.size(1), -1)     # (B, 128, W*H)
        x = x.permute(0, 2, 1)                   # (B, W*H, 128)
        x = self.capsule(x)                      # (B, 10, 16)

        out = torch.sqrt(torch.norm(x, 2, -1))   # (B, 10)
        if y is None:
            return out
        loss = y * torch.relu(0.9 - out)**2 + 0.25*(1 - y)*torch.relu(out - 0.1)**2

        return torch.sum(loss)

