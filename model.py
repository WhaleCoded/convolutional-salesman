import torch
import torch.nn as nn


class TSPConv(nn.Module):
    LARGE_VALUE = 10000.0

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(256, 128, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(128, 64, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 32, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 1, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(1, 1, 7, padding=3),
        )

    def forward(self, x):
        inputs = TSPConv.scrub_inf(x)
        return self.net(inputs)

    def scrub_inf(tensor: torch.Tensor) -> torch.Tensor:
        device = tensor.device
        return torch.where(
            tensor == torch.inf,
            torch.full(tensor.shape, TSPConv.LARGE_VALUE).to(device),
            tensor,
        )
