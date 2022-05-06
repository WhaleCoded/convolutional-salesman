import torch
import torch.nn as nn


class TspConv(nn.Module):
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
        inputs = TspConv.scrub_inf(x)
        return torch.squeeze(self.net(inputs))

    def scrub_inf(tensor: torch.Tensor) -> torch.Tensor:
        device = tensor.device
        return torch.where(
            tensor == torch.inf,
            torch.full(tensor.shape, TspConv.LARGE_VALUE).to(device),
            tensor,
        )


class TspLoss:
    def __init__(self, epsilon: float = 1e-15) -> None:
        self.epsilon = epsilon

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_mask = torch.where(
            (target == torch.inf) | (target == torch.nan), 0, 1
        ).bool()
        prediction_device = prediction.device
        errors = torch.square(target - torch.squeeze(prediction))
        errors = torch.where(
            target_mask, errors, torch.zeros(errors.shape).to(prediction_device)
        )
        return torch.mean(errors) + self.epsilon
