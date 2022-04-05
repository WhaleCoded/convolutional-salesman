import torch
from typing import List
from dataset import TSPDataset
from torch.utils.data import DataLoader
import q_helper
from tqdm import tqdm

NUM_CITIES = 5

dataset = TSPDataset(length=8000, num_cities=NUM_CITIES)
train_set, test_set = dataset.split()

train_loader = DataLoader(dataset=train_set, batch_size=32)
test_loader = DataLoader(dataset=test_set, batch_size=32)


num_without_solution = 0
for batch in tqdm(train_loader):
    cost_matrix = batch["data"]
    costs = q_helper.calculate_greedy_cost(cost_matrix)
    for cost in costs:
        if cost == torch.inf:
            num_without_solution += 1


print(f"There were {num_without_solution} scenarios without solution")