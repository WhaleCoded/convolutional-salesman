from jinja2 import pass_context
import numpy as np
from typing import Tuple, List
import copy
import math

from torch import TensorType
import torch

Q_DISCOUNT_FACTOR = 0.9
NUM_SPLITS = 1

def calculate_q_matrix(
    model: torch.nn.Module, environment_state: torch.Tensor
) -> torch.Tensor:
    cost_matrix, _, previous_move = torch.split(environment_state, 1, dim=1)
    n_cities = cost_matrix.shape[0]
    q_actual = torch.full((n_cities, n_cities), fill_value=torch.inf)

    if any(previous_move):
        _, prev_target = torch.argmax(previous_move)
        for next_target in range(n_cities):
            if cost_matrix[prev_target, next_target] == torch.inf:
                q_actual[prev_target, next_target] = torch.inf
            else:
                q_actual[prev_target, next_target] = calculate_q(
                    model, environment_state, prev_target, next_target
                )
    else:
        for next_origin in range(n_cities):
            for next_target in range(n_cities):
                if cost_matrix[next_origin, next_target] == torch.inf:
                    q_actual[next_origin, next_target] = torch.inf
                else:
                    q_actual[next_origin, next_target] = calculate_q(
                        model, environment_state, next_origin, next_target
                    )
    return q_actual

def update_state(environment_state, best_next_move):
    cost_matrix, current_path, previous_move = torch.split(environment_state, 1, dim=1)
    origin_city, target_city = best_next_move
    previous_move = torch.zeros(cost_matrix.shape)
    previous_move[origin_city, target_city] = 1
    current_path[origin_city, target_city] = 1

    cost_matrix[origin_city] = torch.inf
    cost_matrix[:,target_city] = torch.inf 
    
    return torch.stack(
        [cost_matrix, current_path, previous_move], dim=1
    )



def calculate_q(
    model: torch.nn.Module,
    environment_state: torch.Tensor,
    origin_city: int,
    target_city: int,
) -> float:
    cost_matrix, current_path, previous_move = torch.split(environment_state, 1, dim=1)
    n_cities = cost_matrix.shape[0]
    if torch.sum(current_path) >= (n_cities - 1):
        return 0
    future_costs = torch.inf
    with torch.no_grad():
        predicted_q = model(environment_state)
        moves = get_n_model_moves(predicted_q, environment_state, NUM_SPLITS)
        for move in moves:
            next_environment_state = update_state(environment_state, move)
            potential_cost = calculate_q(
                model, next_environment_state, origin_city, target_city
            )
            if potential_cost < future_costs:
                future_costs = potential_cost
    return cost_matrix[origin_city, target_city] + (future_costs * Q_DISCOUNT_FACTOR)


def get_n_model_moves(
    predicted_q: torch.Tensor,
    environment_state: torch.Tensor,
    num_moves: int = 1,
) -> List[Tuple[int, int]]:
    _, _, previous_move = torch.split(environment_state, 1, dim=1)
    if any(previous_move):
        target_city = 0
    else:
        _, target_city = torch.argmax(previous_move)

    return [(target_city, next_city)  for next_city in torch.argsort(predicted_q[target_city][:num_moves])]

def calculate_greedy_cost(cost_matrix):
    cost = 0

    origin_city = 0
    for _ in range(len(cost_matrix) - 1):
        target_city = torch.argmin(cost_matrix[origin_city])
        cost += cost_matrix[origin_city,target_city]
        cost_matrix[origin_city] = torch.inf
        cost_matrix[:,target_city] = torch.inf
        origin_city = target_city

    return cost

def calculate_model_cost(model, cost_matrix):
    current_path = torch.zeros(cost_matrix.shape)
    previous_move = torch.zeros(cost_matrix.shape)
    environment_state = torch.stack(
                [cost_matrix, current_path, previous_move], dim=1
            ).float()

    cost = 0
    with torch.no_grad():
        for _ in range(len(cost_matrix) - 1):
            q_pred = model(environment_state)
            origin_city, target_city = get_n_model_moves(q_pred, environment_state)[0]
            cost += cost_matrix[origin_city,target_city]
            environment_state = update_state(environment_state, (origin_city, target_city))

    return cost


