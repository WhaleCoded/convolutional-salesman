from typing import Callable, Dict
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm

import q_helper
from data import TspDataset
from model import TspConv, TspLoss

NUM_EPOCHS = 200
NUM_CITIES = 10

model = TspConv()
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
model.to(device)

dataset = TspDataset(length=100, num_cities=NUM_CITIES)
train_set, validation_set = dataset.split()

train_loader = DataLoader(dataset=train_set, batch_size=32)
validation_loader = DataLoader(dataset=validation_set, batch_size=32)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-8, weight_decay=0.3, betas=(0.95, 0.99)
)

loss_fn = TspLoss()


def train_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.optimizer.Optimizer,
) -> torch.Tensor:
    cost_matrix = batch["data"].to(device)
    current_path = torch.zeros(cost_matrix.shape).to(device)
    previous_move = torch.zeros(cost_matrix.shape).to(device)

    # Rather than stepping through the rest of the path each time we take a step
    # we should take all of our steps, and then update the model

    # Although this will reduce our utilization of that particular batch, since
    # we will not update our behavior before moving to the next city, it will speed
    # up how quickly we get to a more novel set of examples. That is more valuable
    # anyways
    for step in range(NUM_CITIES - 1):
        environment_states = torch.stack(
            [cost_matrix, current_path, previous_move], dim=1
        ).float()
        q_preds = model(environment_states)
        q_act = q_helper.calculate_q_matrix(model, environment_states, num_splits=1).to(
            device
        )

        loss = loss_function(q_preds, q_act)
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
            environment_state = q_helper.update_state(environment_state, best_moves[0])
            next_environment_states.append(environment_state)

        if len(next_environment_states) == 0:
            p_bar.update((NUM_CITIES - 1) - step)
            break
        environment_states = torch.stack(next_environment_states, dim=0)


def validation_step(
    model: torch.nn.Module, batch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    cost_matrix = batch["data"].to(device)
    path_cost = q_helper.calculate_model_cost(model, cost_matrix)
    greedy_cost = q_helper.calculate_greedy_cost(cost_matrix)
    return path_cost - greedy_cost


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
    model.train()
    for batch in train_loader:
        loss = train_step(model, batch, optimizer)
        loss_values.append(loss)
        p_bar.set_postfix({"average_loss": f"{np.mean(loss_values[-5:]):.2f}"})
        p_bar.update(1)

    # Evaluate the model
    abs_loss_values = []
    p_bar = tqdm(total=len(), desc="Validation", position=0, leave=True)
    for batch in train_loader:
        with torch.no_grad():
            loss_val = validation_step(model, batch)
            abs_loss_values.append(loss_val)
        p_bar.set_postfix(
            {"difference_from_greedy": np.average(torch.cat(abs_loss_values, 0))}
        )

    # save the model
    if np.average(torch.cat(abs_loss_values, 0)) < curr_best_test:
        print("Saving best model")
        curr_best_test = np.average(torch.cat(abs_loss_values, 0))
        torch.save(model, "results")
