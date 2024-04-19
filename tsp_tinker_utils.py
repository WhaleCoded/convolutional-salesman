from enum import StrEnum
from typing import List, Tuple
import os

import numpy as np


def get_tsp_problem_folders(
    data_path: os.PathLike,
) -> List[Tuple[str, int, os.PathLike]]:
    # Returns the folder name, number_of_cities, and the full path to the folder
    if not os.path.isdir(data_path):
        return []

    tsp_folders = []
    folder_contents = os.listdir(data_path)

    for folder in folder_contents:
        # Regex match tsp_problems_of_*_cities
        if (
            folder.startswith("tsp_problems_of_")
            and folder.endswith("_cities")
            and folder.replace("tsp_problems_of_", "").replace("_cities", "").isdigit()
        ):
            folder_path = os.path.join(data_path, folder)
            num_cities = int(
                folder.replace("tsp_problems_of_", "").replace("_cities", "")
            )
            if os.path.isdir(folder_path):
                tsp_folders.append((folder, num_cities, folder_path))

    return tsp_folders


class TSPAlgorithm(StrEnum):
    LIN_KERNIGHAN = "Lin-Kernighan"
    PSUEDORANDOM = "Pseudorandom"
    NAIVE_HEURISTIC = "Naive Heuristic"
    BANCH_N_BOUND = "Branch and Bound"
    UNKNOWN = "Unknown"


class TSPSolution:
    tot_cost: float
    tour: List[int]
    optimal: bool
    calculation_time: float
    algorithm_name: TSPAlgorithm

    def __init__(
        self,
        tot_cost: float,
        tour: List[int],
        optimal: bool,
        calculation_time: float,
        algorithm_name: TSPAlgorithm,
    ) -> None:
        self.tot_cost = tot_cost
        self.tour = tour
        self.optimal = optimal
        self.calculation_time = calculation_time
        self.algorithm_name = algorithm_name

    @classmethod
    def from_json(cls, json_dict: dict):
        tot_cost = float(json_dict["tot_cost"])
        tour = [int(city_index) for city_index in json_dict["path"]]
        optimal = bool(json_dict["optimal"])
        calculation_time = float(json_dict["calculation_time"])

        algorithm_string = json_dict["algorithm_name"]
        if algorithm_string not in TSPAlgorithm.__members__:
            algorithm_name = TSPAlgorithm.UNKNOWN
        else:
            algorithm_name = TSPAlgorithm(algorithm_string)

        return cls(tot_cost, tour, optimal, calculation_time, algorithm_name)


class TSPProblem:
    city_connections_w_costs: np.ndarray
    num_cities: int
    undirected_edges: bool

    def __init__(self, city_connections_w_costs: np.ndarray, undirected_edges: bool):
        self.city_connections_w_costs = city_connections_w_costs
        self.num_cities = city_connections_w_costs.shape[0]
        self.undirected_edges = undirected_edges

    @classmethod
    def from_json(cls, json_dict: dict):
        num_cities = int(json_dict["num_cities"])
        undirected_edges = json_dict["undirected"]

        if undirected_edges:
            # Only the upper triangle of the matrix store as a 1D array
            city_connections_w_costs = np.zeros((num_cities, num_cities))
            upper_triangle = json_dict["upper_triangle_edges"]

            curr_row = 0
            curr_col = 0
            for edge_cost in upper_triangle:
                city_connections_w_costs[curr_row, curr_col] = edge_cost
                city_connections_w_costs[curr_col, curr_row] = edge_cost
                curr_col += 1
                if curr_col == num_cities:
                    curr_row += 1
                    curr_col = curr_row
        else:
            city_connections_w_costs = np.array(json_dict["city_connections_w_costs"])

        return cls(city_connections_w_costs, undirected_edges)


class TSPPackage:
    problem: TSPProblem
    solutions: List[TSPSolution]
    uuid: str

    def __init__(
        self, uuid: str, problem: TSPProblem, solutions: List[TSPSolution]
    ) -> None:
        self.uuid = uuid
        self.problem = problem
        self.solutions = solutions
        best_cost = min([sol.tot_cost for sol in solutions])
        self.best_solution = next(
            (sol for sol in solutions if sol.tot_cost == best_cost), None
        )

    def get_best_solution(self) -> TSPSolution:
        return self.best_solution

    @classmethod
    def from_json(cls, json_dict: dict):
        uuid = json_dict["uuid"]
        problem = TSPProblem.from_json(json_dict["problem_data"])
        solutions = [TSPSolution.from_json(sol) for sol in json_dict["solutions"]]
        return cls(uuid, problem, solutions)
