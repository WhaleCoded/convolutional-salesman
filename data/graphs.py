import torch

IMPOSSIBLE_VALUE = 10.0


def generate_solvable_graph(
    num_cities: int = 10, connection_proportion: float = 0.5
) -> torch.Tensor:
    # Generate random weights between -1 and 1 for all possible edges
    weights = (torch.rand((num_cities, num_cities)) * 2) - 1
    # Generate a random mask for the 'extra' connections given likelihood
    connections = torch.rand((num_cities, num_cities)) < connection_proportion
    # Generate a valid path mask and ensure that cities are not self-connected
    path = torch.randperm(num_cities)
    prev_city = path[0]
    connections[prev_city, prev_city] = 0
    for city in path[1:]:
        connections[prev_city, city] = 1
        connections[city, city] = 0
        prev_city = city
    # Create final graph
    infinities = torch.ones((num_cities, num_cities)) * IMPOSSIBLE_VALUE
    return torch.stack([torch.where(connections, weights, infinities), connections])
