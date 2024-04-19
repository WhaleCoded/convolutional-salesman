import os
import random
import json
from typing import List, Tuple, Optional, Dict, Set

import numpy as np
import torch
from multiprocessing import Pool, cpu_count
from tqdm.notebook import tqdm

from tsp_tinker_utils import TSPPackage, get_tsp_problem_folders


def _normalize_np_array(arr: np.ndarray) -> np.ndarray:
    max_val = np.max(arr)
    min_val = np.min(arr)
    return (arr - min_val) / (max_val - min_val) * 2 - 1


def _sample_path_variations_from_problem(
    city_connections_w_costs: np.ndarray,
    edge_matrix: np.ndarray,
    edges: List[Tuple[int, int]],
    num_samples: int,
) -> Tuple[List[List[Tuple[int, int]]], np.ndarray, np.ndarray]:
    num_cities = city_connections_w_costs.shape[0]

    batch_selected_edges = []
    num_edges_lb = 0
    num_edges_ub = num_cities - 2
    for i in range(num_samples):
        num_edges_to_include = random.randint(num_edges_lb, num_edges_ub)
        direction = bool(random.randint(0, 1))

        if num_edges_to_include == 0:
            num_edges_lb = 1

        random.shuffle(edges)
        if direction:
            batch_selected_edges.append(edges[:num_edges_to_include])
        else:
            reversed_edges = [
                (out_c, in_c) for in_c, out_c in edges[:num_edges_to_include]
            ]
            batch_selected_edges.append(reversed_edges)

    target = edge_matrix

    # Normalize the cost matrix between -1 and 1
    max_val = np.max(city_connections_w_costs)
    min_val = np.min(city_connections_w_costs)
    city_connections_w_costs = (city_connections_w_costs - min_val) / (
        max_val - min_val
    )

    return (batch_selected_edges, target, _normalize_np_array(city_connections_w_costs))


def _load_and_process_tsp_problem(
    file_path: os.PathLike,
    num_path_variations_per_example: int,
) -> Tuple[Tuple[List[List[Tuple[int, int]]], np.ndarray, np.ndarray], str]:
    with open(file_path, "r") as file:
        json_data = json.load(file)

    packaged_problem = TSPPackage.from_json(json_data)
    problem = packaged_problem.problem
    best_solution = packaged_problem.best_solution

    # Convert the path to edge matrix
    edge_matrix = np.zeros((problem.num_cities, problem.num_cities), dtype=np.float32)
    edges = []
    previous_city = 0
    for city in best_solution.tour:
        edge_matrix[previous_city, city] = 1
        edges.append((previous_city, city))
        previous_city = city

    return (
        _sample_path_variations_from_problem(
            problem.city_connections_w_costs,
            edge_matrix,
            edges,
            num_path_variations_per_example,
        ),
        packaged_problem.uuid,
    )


class _NumVarCopier(object):
    def __init__(self, num_var):
        self.num_var = num_var

    def __call__(self, x):
        return _load_and_process_tsp_problem(x, self.num_var)


class TSPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ):
        super(TSPDataset, self).__init__()
        self.data = data

    @classmethod
    def from_disk(
        cls,
        data_folder_path: os.PathLike,
        num_path_variations: int,
        problem_size_lower_bound: Optional[int] = None,
        problem_size_upper_bound: Optional[int] = None,
        undirected_only: Optional[bool] = True,
        max_workers: int = cpu_count() - 1,
    ):
        new_instance = cls([])
        data = new_instance._multi_threaded_load(
            data_folder_path,
            num_path_variations,
            problem_size_lower_bound,
            problem_size_upper_bound,
            max_workers,
        )
        new_instance.data = data

        return new_instance

    def _multi_threaded_load(
        self,
        data_folder_path: os.PathLike,
        num_path_variations: int,
        problem_size_lower_bound: Optional[int],
        problem_size_upper_bound: Optional[int],
        max_workers: int,
    ):
        possible_folders = get_tsp_problem_folders(data_folder_path)
        problem_file_paths = []
        for _, problem_size, folder_path in possible_folders:
            if (
                problem_size_lower_bound is None
                or problem_size >= problem_size_lower_bound
            ) and (
                problem_size_upper_bound is None
                or problem_size <= problem_size_upper_bound
            ):
                # Add all files in the folder to the list of files to load
                for file in os.listdir(folder_path):
                    if file.endswith(".json"):
                        problem_file_paths.append(os.path.join(folder_path, file))

        formatted_data = []
        var_copier = _NumVarCopier(num_path_variations)
        with Pool(processes=max_workers) as worker_pool:
            with tqdm(
                total=len(problem_file_paths), desc="Loading data from disk..."
            ) as p_bar:
                for data in worker_pool.imap_unordered(
                    var_copier,
                    problem_file_paths,
                    chunksize=10,
                ):
                    p_bar.update(1)
                    formatted_data.append(data)

        return formatted_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (path_variation_edges, edge_matrix, cost_matrix), _ = self.data[idx]
        num_variations = len(path_variation_edges)

        target = torch.tensor(edge_matrix, dtype=torch.float32)
        target = target.repeat(num_variations, 1, 1)

        batch = np.zeros(
            (num_variations, 2, cost_matrix.shape[0], cost_matrix.shape[1])
        )
        for i, path in enumerate(path_variation_edges):
            for in_c, out_c in path:
                batch[i, 1, in_c, out_c] = 1

            batch[i, 0] = cost_matrix

        return torch.tensor(batch, dtype=torch.float32), target

    def split(self, ration: float = 0.1):
        num_to_take = int(len(self) * ration)
        random.shuffle(self.data)

        split_data = self.data[:num_to_take]
        self.data = self.data[num_to_take:]

        return TSPDataset(split_data)

    def split_by_uuids(self, uuids: Set[str]):
        split_data = []
        remaining_data = []
        for data, uuid in self.data:
            if uuid in uuids:
                split_data.append((data, uuid))
            else:
                remaining_data.append((data, uuid))

        self.data = remaining_data

        return TSPDataset(split_data)

    def stratified_split(self, num_examples: int):
        num_per_size = {}
        for data, _ in self.data:
            num_cities = data[2].shape[-1]
            if num_cities not in num_per_size:
                num_per_size[num_cities] = 0

            num_per_size[num_cities] += 1

        num_problem_sizes = len(num_per_size.keys())
        num_to_select = num_examples // num_problem_sizes
        remaining = num_examples % num_problem_sizes
        selected_data = []
        selected_sizes = {k: num_to_select for k in num_per_size.keys()}

        # Make selection random
        random.shuffle(self.data)
        staying_data = []
        for _ in range(len(self.data)):
            data = self.data.pop()
            data_prob_size = data[0][2].shape[-1]

            if selected_sizes[data_prob_size] > 0:
                selected_data.append(data)
                selected_sizes[data_prob_size] -= 1
            else:
                staying_data.append(data)

        assert len(self.data) == 0

        for num_to_select in selected_sizes.values():
            remaining += num_to_select

        selected_data += staying_data[:remaining]
        self.data = staying_data[remaining:]

        return TSPDataset(selected_data)

    def get_uuids(self) -> List[str]:
        return [uuid for _, uuid in self.data]
