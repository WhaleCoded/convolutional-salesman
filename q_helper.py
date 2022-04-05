import numpy as np
from typing import Tuple, List
import copy
import math

from torch import TensorType
import torch

IMPOSSIBLE_VALUE = 1000000000000.0

def calc_q_actual(model: torch.nn.Module, cost_matrix: torch.Tensor, curr_path: torch.Tensor):
    with torch.no_grad():
                


def convert_edge_list_to_matrix(curr_path: Tuple[int, int], num_cities) -> np.ndarray:
    final_matrix = np.zeros((num_cities, num_cities))
    for edge in curr_path:
        final_matrix[edge[0]][edge[1]] = 1

    return final_matrix

def convert_matrix_to_edge_list(path_matrix: np.ndarray) -> Tuple[int, int]:
    edge_list = []
    for i in range(len(path_matrix)):
        for x in range(len(path_matrix)):
            if path_matrix[i][x] == 1:
                edge_list.append((i,x))

    return edge_list

def expand_state(curr_matrix: np.ndarray, curr_lower_bound: float, curr_path: List[Tuple[int,int]]) -> List[Tuple[np.ndarray, float, List[Tuple[int,int]]]]:
		new_states = []	

		#we need to go from this city to the next city
		if len(curr_path) == 0:
			next_city_index = 0
		else:
			next_city_index = curr_path[len(curr_path) -1 ][1]
		possible_moves = get_next_possible_locations(curr_matrix, next_city_index)

		#get possible moves from this city
		for edge in possible_moves:
			temp = copy.deepcopy(curr_matrix)
			temp, additional_cost = expand_on_location(edge, temp)
			new_path = copy.deepcopy(curr_path)
			new_path.append(edge)
			new_states.append((temp, curr_lower_bound + additional_cost, new_path))

		return new_states

# get the coordinates of each zero in the cost matrix
	#O(n^2) time and O(n) space
def get_next_possible_locations(curr_matrix: np.ndarray, next_city_index: int) -> List[Tuple[int,int]]:
    possible_locations = []

    #O(n^2) time and O(n) space
    for x in range(len(curr_matrix)):
        if curr_matrix[next_city_index][x] != IMPOSSIBLE_VALUE:
            possible_locations.append((next_city_index, x))

    return possible_locations

#calculates a new state and lower_bound based on a new move
# time complexity is O(n^2) and space is O(n^2)	
def expand_on_location(next_move: Tuple[int,int], curr_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    #replace row and column of the next_move wiht infinty
    #O(n)
    temp = copy.deepcopy(curr_matrix)
    move_cost = temp[next_move[0]][next_move[1]]
    temp[next_move[1]][next_move[0]] = IMPOSSIBLE_VALUE
    temp[next_move[0]] = [IMPOSSIBLE_VALUE for _ in range(len(curr_matrix[0]))]
    for i in range(len(curr_matrix)):
        temp[i][next_move[1]] = IMPOSSIBLE_VALUE

    #adjust the matrix
    new_cost_matrix, additional_cost = solve_cost_matrix(temp)
    return (new_cost_matrix, additional_cost + move_cost)

#this function makes sure there is a 0 in every column or 
#row where the whole row is not infinity
#time complexity is O(n^2) and space is O(n^2)
def solve_cost_matrix(curr_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    additional_cost = 0

    #adjust rows
    #O(n^2)
    for i in range(len(curr_matrix)):
        
        curr_min = curr_matrix[i][0]
        for x in range(len(curr_matrix[i])): 
            if curr_matrix[i][x] < curr_min:
                curr_min = curr_matrix[i][x]

        #if the row is all inf or has a 0 we dont have to adjust
        if curr_min != IMPOSSIBLE_VALUE and curr_min != 0:
            #subtract the minimum from each item in the row
            curr_matrix[i] = [curr_matrix[i][x] - curr_min for x in range(len(curr_matrix[i]))]
            additional_cost += curr_min

    #adjust columns
    #O(n^2)
    for i in range(len(curr_matrix[0])):

        curr_min = curr_matrix[0][i]
        for x in range(len(curr_matrix)):
            if curr_matrix[x][i] < curr_min:
                curr_min = curr_matrix[x][i]

        #if the column is all inf or has a 0 we dont have to adjust
        if curr_min != IMPOSSIBLE_VALUE and curr_min != 0:
            #subtract the minimum from each item in the column
            additional_cost += curr_min
            for x in range(len(curr_matrix)):
                curr_matrix[x][i] = curr_matrix[x][i] - curr_min

    return curr_matrix, additional_cost