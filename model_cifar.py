import torch.nn as nn
import torch.nn.functional as F


class CIFAR_CNN(nn.Module):
    """
    3-layer CNN for CIFAR-10 (3-channel 32×32 RGB input, 10 classes).
    Larger than the MNIST CNN to handle the increased complexity.

    Architecture:
        Conv(3→32) → ReLU → Pool    # 32×32 → 16×16
        Conv(32→64) → ReLU → Pool   # 16×16 → 8×8
        Conv(64→128) → ReLU         # 8×8 (no pool)
        FC(8192→256) → ReLU
        FC(256→10)
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
