from re import M
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import numpy as np
import q_helper

from dataset import TSPDataset
from tqdm import tqdm

NUM_EPOCHS = 10
NUM_CITIES = 10

model = nn.Sequential(
    nn.Conv2d(3, 16, 5, padding=2),
    nn.ReLU(),
    nn.Conv2d(16, 32, 5, padding=2),
    nn.ReLU(),
    nn.Conv2d(32, 64, 5, padding=2),
    nn.ReLU(),
    nn.Conv2d(64, 128, 5, padding=2),
    nn.ReLU(),
    nn.Conv2d(128, 64, 5, padding=2),
    nn.ReLU(),
    nn.Conv2d(64, 32, 5, padding=2),
    nn.ReLU(),
    nn.Conv2d(32, 16, 5, padding=2),
    nn.ReLU(),
    nn.Conv2d(16, 1, 5, padding=2),
)
# device = "cuda:0"
# model.to(device=device)

dataset = TSPDataset(length=500, num_cities=NUM_CITIES)
train_set, test_set = dataset.split()

train_loader = DataLoader(dataset=train_set, batch_size=32)
test_loader = DataLoader(dataset=test_set, batch_size=32)


def loss_fn(prediction, target):
    errors = torch.square(target - prediction)
    errors = torch.where(errors == torch.inf, torch.zeros(errors.shape), errors)
    return torch.mean(errors)


optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-3, weight_decay=0.3, betas=(0.95, 0.99)
)


curr_best_test = torch.inf
for epoch in range(NUM_EPOCHS):
    print(f"--- Epoch {epoch} ---")
    p_bar = tqdm(total=len(train_loader) * (NUM_CITIES - 1), desc="Training")

    loss_values = []
    for batch in train_loader:
        cost_matrix = batch["data"]
        current_path = torch.zeros(cost_matrix.shape)
        previous_move = torch.zeros(cost_matrix.shape)

        for step in range(NUM_CITIES - 1):
            environment_state = torch.stack(
                [cost_matrix, current_path, previous_move], dim=1
            ).float()
            q_pred = model(environment_state)
            q_act = q_helper.calculate_q_matrix(model, environment_state)

            loss = loss_fn(q_pred, q_act)
            loss_values.append(loss.detach().item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            environment_state = q_helper.update_state(q_pred, environment_state)

            p_bar.set_postfix({"average_loss": np.average(loss_values[-20:])})
            p_bar.update(1)

    # test the data
    abs_loss_values = []
    t_p_bar = tqdm(train_loader, desc="Validation")
    for batch in t_p_bar:
        with torch.no_grad():
            path_cost = q_helper.calculate_model_cost(model, batch["data"])
            greedy_cost = q_helper.calculate_greedy_cost(batch["data"])
            abs_loss_values.append(path_cost - greedy_cost)
            p_bar.set_postfix(
                {"absolute_loss": np.average(torch.cat(abs_loss_values, 0))}
            )

    # save the model
    if np.average(torch.cat(abs_loss_values, 0)) < curr_best_test:
        print("We saved a model")
        curr_best_test = np.average(torch.cat(abs_loss_values, 0))
        torch.save(model, "results")
