import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import numpy as np
import q_helper

from data.dataset import TSPDataset
from tqdm import tqdm

NUM_EPOCHS = 200
NUM_CITIES = 10


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


model = TSPConv()
device = "cuda:0"
model.to(device)

dataset = TSPDataset(length=100, num_cities=NUM_CITIES)
train_set, test_set = dataset.split()

train_loader = DataLoader(dataset=train_set, batch_size=32)
test_loader = DataLoader(dataset=test_set, batch_size=32)


def loss_fn(prediction, target):
    target_mask = torch.where(
        (target == torch.inf) | (target == torch.nan), 0, 1
    ).bool()
    prediction_device = prediction.device
    errors = torch.square(target - torch.squeeze(prediction))
    errors = torch.where(
        target_mask, errors, torch.zeros(errors.shape).to(prediction_device)
    )
    return torch.mean(errors) + 1e-15


optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-8, weight_decay=0.3, betas=(0.95, 0.99)
)

curr_best_test = torch.inf
for epoch in range(NUM_EPOCHS):
    print(f"--- Epoch {epoch}/{NUM_EPOCHS} ---")
    p_bar = tqdm(
        total=len(train_loader) * (NUM_CITIES - 1),
        desc="Training",
        position=0,
        leave=True,
    )

    loss_values = []
    for batch in train_loader:
        cost_matrix = batch["data"].to(device)
        current_path = torch.zeros(cost_matrix.shape).to(device)
        previous_move = torch.zeros(cost_matrix.shape).to(device)

        for step in range(NUM_CITIES - 1):
            environment_states = torch.stack(
                [cost_matrix, current_path, previous_move], dim=1
            ).float()
            q_preds = model(environment_states)
            q_act = q_helper.calculate_q_matrix(
                model, environment_states, num_splits=1
            ).to(device)

            loss = loss_fn(q_preds, q_act)
            loss_values.append(loss.detach().item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            next_environment_states = []
            for q_pred, environment_state in zip(q_preds, environment_states):
                q_pred, environment_state = torch.squeeze(q_pred), torch.squeeze(
                    environment_state
                )
                best_moves = q_helper.get_n_model_moves(q_pred, environment_state)
                if len(best_moves) == 0:
                    continue
                environment_state = q_helper.update_state(
                    environment_state, best_moves[0]
                )
                next_environment_states.append(environment_state)
            p_bar.set_postfix({"average_loss": f"{np.mean(loss_values[-5:]):.2f}"})
            if len(next_environment_states) == 0:
                p_bar.update((NUM_CITIES - 1) - step)
                break
            environment_states = torch.stack(next_environment_states, dim=0)
            p_bar.update(1)

    # test the data
    abs_loss_values = []
    p_bar = tqdm(train_loader, desc="Validation", position=0, leave=True)
    for batch in p_bar:
        with torch.no_grad():
            cost_matrix = batch["data"].to(device)
            path_cost = q_helper.calculate_model_cost(model, cost_matrix)
            greedy_cost = q_helper.calculate_greedy_cost(cost_matrix)
            abs_loss_values.append(path_cost - greedy_cost)
            p_bar.set_postfix(
                {"absolute_loss": np.average(torch.cat(abs_loss_values, 0))}
            )

    # save the model
    if np.average(torch.cat(abs_loss_values, 0)) < curr_best_test:
        print("We saved a model")
        curr_best_test = np.average(torch.cat(abs_loss_values, 0))
        torch.save(model, "results")
