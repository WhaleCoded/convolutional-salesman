from typing import Dict
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm

import q_helper
from data import TspDataset
from model import TspConv

NUM_EPOCHS = 200
NUM_CITIES = 10

model = TspConv()
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
model.to(device)

dataset = TspDataset(length=100000, num_cities=NUM_CITIES)
train_set, validation_set = dataset.split()

train_loader = DataLoader(dataset=train_set, batch_size=32)
validation_loader = DataLoader(dataset=validation_set, batch_size=32)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-8, weight_decay=0.3, betas=(0.95, 0.99)
)

loss_fn = torch.nn.MSELoss()


def train_step(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    cost_matrix = batch["data"][:, 0].to(device)
    connections = batch["data"][:, 1].to(device)
    current_path = torch.zeros(cost_matrix.shape).to(device)

    # Rather than stepping through the rest of the path each time we take a step
    # we should take all of our steps, and then update the model

    # Although this will reduce our utilization of that particular batch, since
    # we will not update our behavior before moving to the next city, it will speed
    # up how quickly we get to a more novel set of examples. That is more valuable
    # anyways

    batch_size = cost_matrix.shape[0]
    current_city = torch.randint(0, NUM_CITIES, (batch_size,))
    city_indices = torch.arange(batch_size)
    path_cost = torch.zeros((batch_size,))
    cost_predictions = []
    for _ in range(NUM_CITIES - 1):
        environment_state = torch.stack([cost_matrix, connections, current_path], dim=1)
        q_preds = model(environment_state)
        city_predictions = torch.squeeze(q_preds[city_indices, current_city, :])

        # Select the best non-inf value
        adjusted_predictions = torch.where(
            cost_matrix[city_indices, current_city, :] < 1.0,
            city_predictions,
            cost_matrix[city_indices, current_city, :],
        )
        best_predicted_cost, selected_destination = torch.min(
            adjusted_predictions, dim=1
        )
        cost_predictions.append(best_predicted_cost)
        path_cost += cost_matrix[city_indices, current_city, selected_destination]
        current_city = selected_destination

        # Update state before we move on
        current_path[city_indices, current_city, selected_destination] = 1
        connections[city_indices, current_city, selected_destination] = 0
        cost_matrix[city_indices, current_city, selected_destination] = 1.5

    total_loss = 0
    for cost_prediction in cost_predictions:
        total_loss += loss_fn(cost_prediction, path_cost)

    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return total_loss.detach().item()


def validation_step(
    model: torch.nn.Module, batch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    cost_matrix = batch["data"][:, 0].to(device)
    connections = batch["data"][:, 1].to(device)
    current_path = torch.zeros(cost_matrix.shape).to(device)

    # path_cost = q_helper.calculate_model_cost(model, cost_matrix)
    greedy_cost = q_helper.calculate_greedy_cost(cost_matrix)
    print(greedy_cost)
    return path_cost - greedy_cost


curr_best_test = torch.inf
for epoch in range(NUM_EPOCHS):
    print(f"--- Epoch {epoch}/{NUM_EPOCHS} ---")
    p_bar = tqdm(
        total=len(train_loader),
        desc="Training",
        position=0,
        leave=True,
    )

    loss_values = []
    model.train()
    for batch in train_loader:
        loss = train_step(model, batch, optimizer)
        loss_values.append(loss)
        p_bar.set_postfix({"average_loss": f"{np.mean(loss_values[-150:]):.2f}"})
        p_bar.update(1)

    # Evaluate the model
    # abs_loss_values = []
    # p_bar = tqdm(
    #     total=len(validation_loader), desc="Validation", position=0, leave=True
    # )
    # for batch in validation_loader:
    #     with torch.no_grad():
    #         loss_val = validation_step(model, batch)
    #         abs_loss_values.append(loss_val)
    #     p_bar.set_postfix(
    #         {"difference_from_greedy": np.average(torch.cat(abs_loss_values, 0))}
    #     )
    #     p_bar.update(1)

    # # save the model
    # if np.average(torch.cat(abs_loss_values, 0)) < curr_best_test:
    #     print("Saving best model")
    #     curr_best_test = np.average(torch.cat(abs_loss_values, 0))
    #     torch.save(model, "results")
