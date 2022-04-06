from typing import DefaultDict, Tuple, List
import torch

from utils import unravel_index
import numpy as np
import random

LARGE_VALUE = 100.0
Q_DISCOUNT_FACTOR = 0.9


def calculate_q_matrix(
    model: torch.nn.Module,
    environment_states: torch.Tensor,
    num_splits: int = 2,
) -> torch.Tensor:
    q_actuals = []
    for environment_state in environment_states:
        cost_matrix, _, previous_move = torch.split(environment_state, 1, dim=0)
        cost_matrix, previous_move = torch.squeeze(cost_matrix), torch.squeeze(
            previous_move
        )
        n_cities = cost_matrix.shape[0]
        q_actual = torch.full((n_cities, n_cities), fill_value=torch.inf)
        if torch.sum(previous_move) > 0:
            _, prev_target = torch.argmax(previous_move)
            for next_target in range(n_cities):
                if next_target == prev_target:
                    continue
                if cost_matrix[prev_target, next_target] == torch.inf:
                    q_actual[prev_target, next_target] = torch.inf
                else:
                    q_actual[prev_target, next_target] = calculate_q(
                        model,
                        environment_state,
                        prev_target,
                        next_target,
                        num_splits=num_splits,
                    )
        else:
            for next_origin in range(n_cities):
                for next_target in range(n_cities):
                    if next_origin == next_target:
                        continue
                    if cost_matrix[next_origin, next_target] == torch.inf:
                        q_actual[next_origin, next_target] = torch.inf
                    else:
                        q_actual[next_origin, next_target] = calculate_q(
                            model,
                            environment_state,
                            next_origin,
                            next_target,
                            num_splits=num_splits,
                        )
        q_actuals.append(q_actual)
    return torch.stack(q_actuals, dim=0)


def update_state(environment_state, best_next_move):
    cost_matrix, current_path, previous_move = torch.split(environment_state, 1, dim=0)
    cost_matrix, current_path, previous_move = (
        torch.squeeze(cost_matrix),
        torch.squeeze(current_path),
        torch.squeeze(previous_move),
    )
    cost_matrix, current_path = cost_matrix.clone(), current_path.clone()
    origin_city, target_city = best_next_move
    device = cost_matrix.device
    previous_move = torch.zeros(cost_matrix.shape).to(device)
    previous_move[origin_city, target_city] = 1
    current_path[origin_city, target_city] = 1

    cost_matrix[origin_city] = torch.inf
    cost_matrix[:, target_city] = torch.inf

    return torch.stack([cost_matrix, current_path, previous_move], dim=0)


def calculate_q(
    model: torch.nn.Module,
    environment_state: torch.Tensor,
    origin_city: int,
    target_city: int,
    num_splits: int = 2,
) -> float:
    cost_matrix, current_path, _ = torch.split(environment_state, 1, dim=0)
    cost_matrix, current_path = torch.squeeze(cost_matrix), torch.squeeze(current_path)
    n_cities = cost_matrix.shape[0]
    if torch.sum(current_path) >= (n_cities - 1):
        return 0
    future_costs = torch.inf
    with torch.no_grad():
        predicted_q = model(torch.unsqueeze(environment_state, dim=0))
        moves = get_n_model_moves(
            torch.squeeze(predicted_q), environment_state, num_splits
        )
        for move in moves:
            next_environment_state = update_state(environment_state, move)
            potential_cost = calculate_q(
                model,
                next_environment_state,
                origin_city,
                target_city,
                num_splits=num_splits,
            )
            if potential_cost < future_costs:
                future_costs = potential_cost
    if future_costs == torch.inf:
        future_costs = LARGE_VALUE
    return cost_matrix[origin_city, target_city] + (future_costs * Q_DISCOUNT_FACTOR)


def get_n_model_moves(
    predicted_q: torch.Tensor,
    environment_state: torch.Tensor,
    num_moves: int = 1,
) -> List[Tuple[int, int]]:
    cost_matrix, _, previous_move = torch.split(environment_state, 1, dim=0)
    cost_matrix, previous_move = torch.squeeze(cost_matrix), torch.squeeze(
        previous_move
    )
    if torch.sum(previous_move) > 0:
        _, origin_city = unravel_index(torch.argmax(previous_move), previous_move.shape)
    else:
        origin_city = 0

    sorted_moves = torch.argsort(predicted_q[origin_city])
    valid_moves = []
    for target_city in sorted_moves:
        if not cost_matrix[origin_city, target_city] == torch.inf:
            valid_moves.append((origin_city, target_city.item()))
    return valid_moves[:num_moves]


def calculate_greedy_cost(cost_matrix: torch.Tensor) -> List[float]:
    costs = [0 for _ in cost_matrix]
    for idx, cost_matrix in enumerate(cost_matrix):
        origin_city = random.randint(0, len(cost_matrix) - 1)
        cities_visited = 0
        visited_cities = DefaultDict(int)
        for _ in range(len(cost_matrix) - 1):
            visited_cities[int(origin_city)] = 1
            cities_visited += 1
            target_city = torch.argmin(cost_matrix[origin_city])
            if visited_cities[int(target_city)] == 1 and cities_visited != len(cost_matrix):
                for city_index in range(len(cost_matrix)):
                    if cost_matrix[origin_city, city_index] != torch.inf and visited_cities[city_index] == 0:
                        target_city = city_index
                        break
            costs[idx] += cost_matrix[origin_city, target_city].to("cpu").item()
            if costs[idx] == torch.inf:
                temp = "tsop"
            cost_matrix[origin_city] = torch.inf
            cost_matrix[:, target_city] = torch.inf
            origin_city = target_city
    return torch.Tensor(costs)

def calculate_geek_greedy_cost(cost_matrix: torch.Tensor) -> List[float]:
    costs = [0 for _ in cost_matrix]
    for idx, cost_matrix in enumerate(cost_matrix):
        sum = 0
        counter = 0
        j = 0
        i = 0
        min = torch.inf
        visitedRouteList = DefaultDict(int)
    
        # Starting from the 0th indexed
        # city i.e., the first city
        visitedRouteList[0] = 1
        route = [0] * len(cost_matrix)
    
        # Traverse the adjacency
        # matrix tsp[][]
        while i < len(cost_matrix) and j < len(cost_matrix[i]):
    
            # Corner of the Matrix
            if counter >= len(cost_matrix[i]) - 1:
                break
    
            # If this path is unvisited then
            # and if the cost is less then
            # update the cost
            if j != i and (visitedRouteList[j] == 0):
                if cost_matrix[i][j] < min:
                    min = cost_matrix[i][j]
                    route[counter] = j + 1
    
            j += 1
    
            # Check all paths from the
            # ith indexed city
            if j == len(cost_matrix[i]):
                sum += min
                min = torch.inf
                visitedRouteList[route[counter] - 1] = 1
                j = 0
                i = route[counter] - 1
                counter += 1
    
        # Update the ending city in array
        # from city which was last visited
        i = route[counter - 1] - 1
    
        for j in range(len(cost_matrix)):
    
            if (i != j) and cost_matrix[i][j] < min:
                min = cost_matrix[i][j]
                route[counter] = j + 1
    
        sum += min
        costs[idx] = sum

    return costs

def calculate_random_cost(cost_matrix: torch.Tensor) -> List[float]:
    costs = [0 for _ in cost_matrix]
    for idx, cost_matrix in enumerate(cost_matrix):
        tour_not_found = True
        perm = np.random.permutation( len(cost_matrix) )
        while tour_not_found:
            #check if perm is valid
            cost = 0
            origin_city = None
            for target_city in perm:
                if origin_city != None:
                    cost += cost_matrix[origin_city, target_city]
                origin_city = target_city

            if cost != torch.inf:
                tour_not_found = False
                costs[idx] = cost
        

    return torch.Tensor(costs)

def calculate_model_cost(
    model: torch.nn.Module, cost_matrices: torch.Tensor
) -> List[float]:
    device = cost_matrices.device
    current_path = torch.zeros(cost_matrices.shape).to(device)
    previous_move = torch.zeros(cost_matrices.shape).to(device)
    environment_states = torch.stack(
        [cost_matrices, current_path, previous_move], dim=1
    ).float()

    costs = [0 for _ in environment_states]
    with torch.no_grad():
        for _ in range(len(cost_matrices) - 1):
            q_preds = model(environment_states)
            next_environment_states = []
            for idx, (q_pred, environment_state) in enumerate(
                zip(q_preds, environment_states)
            ):
                q_pred, environment_state = (
                    torch.squeeze(q_pred),
                    torch.squeeze(environment_state),
                )
                cost_matrix, _, _ = torch.split(environment_state, 1, dim=0)
                cost_matrix = torch.squeeze(cost_matrix)
                moves = get_n_model_moves(q_pred, environment_state)
                if len(moves) == 0:
                    costs[idx] = torch.inf
                    continue
                origin_city, target_city = moves[0]
                costs[idx] = costs[idx] + cost_matrix[origin_city, target_city]
                environment_state = update_state(
                    environment_state, (origin_city, target_city)
                )
                next_environment_states.append(environment_state)
            if len(next_environment_states) == 0:
                break
            environment_states = torch.stack(next_environment_states, dim=0)
    return torch.Tensor(costs)
