from typing import List
import torch
import torch.nn as nn


class ConvolutionalSalesmanNet(nn.Module):
    def __init__(self, kernel_size: int = 5, padding: int = 2):
        super(ConvolutionalSalesmanNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(4),
            nn.GELU(),
            nn.Conv2d(4, 8, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # 10 layers down
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # # 20 layers down, head back up
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm2d(256),
            # nn.GELU(),
            # 10 layers up, 10 layers to go
            nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 8, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 4, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(4),
            nn.GELU(),
            nn.Conv2d(4, 2, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(2),
            nn.GELU(),
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        preds: torch.Tensor = self.net(x)
        return preds.squeeze()

    @property
    def device(self):
        return next(self.parameters()).device


def construct_path(model: ConvolutionalSalesmanNet, connection_costs: torch.Tensor):
    model.eval()

    num_cities = connection_costs.shape[0]
    available_cities = [i for i in range(1, num_cities)]
    curr_city = 0

    connection_costs = connection_costs.to(model.device)
    constructed_path = torch.zeros(num_cities, num_cities).to(model.device)
    for _ in range(num_cities - 1):
        batch = torch.zeros(1, 2, num_cities, num_cities).to(model.device)
        batch[0, 0] = connection_costs
        batch[0, 1] = constructed_path

        edge_preds = model(batch)

        # Get best edge given available cities
        next_city = available_cities[
            torch.argmax(edge_preds[curr_city, available_cities])
        ]
        available_cities.remove(next_city)
        constructed_path[curr_city, next_city] = 1
        curr_city = next_city

    # Close the loop
    constructed_path[curr_city, 0] = 1

    return constructed_path


def validate_path(constructed_path: torch.Tensor) -> List[int]:
    num_cities = constructed_path.shape[0]
    visited_cities = set()
    curr_city = 0
    path = [curr_city]

    while curr_city not in visited_cities:
        visited_cities.add(curr_city)
        next_city = torch.argmax(constructed_path[curr_city, :]).item()
        path.append(next_city)
        curr_city = next_city

    assert len(path) == num_cities + 1

    return path


def calc_path_cost(
    constructed_path: torch.Tensor, connection_costs: torch.Tensor
) -> float:
    return torch.sum(constructed_path * connection_costs).item()


def calc_path_metric(
    generated_path: torch.Tensor,
    target_path: torch.Tensor,
    connection_costs: torch.Tensor,
) -> float:
    # De normalize the paths and connections
    if torch.min(connection_costs) < 0.0:
        connection_costs = (connection_costs + 1) / 2

    if torch.min(generated_path) < 0.0:
        generated_path = (generated_path + 1) / 2

    if torch.min(target_path) < 0.0:
        target_path = (target_path + 1) / 2

    generated_cost = calc_path_cost(generated_path, connection_costs)
    target_cost = calc_path_cost(target_path, connection_costs)

    cost_diff = generated_cost - target_cost

    return cost_diff / target_cost
