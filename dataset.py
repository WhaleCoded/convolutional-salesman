from cProfile import label
from urllib.parse import _NetlocResultMixinStr
from rootflow.datasets.base.generator import GeneratorDataset
from rootflow.datasets.base.item import DataItem
from scenario_gen.tsp_gen_master import TSPMaster
from typing import Tuple, Iterator
import numpy as np

class TSPDataset(GeneratorDataset):
    def __init__(self, length: int, num_cities: int = 1000) -> None:
        self.num_cities = num_cities
        self.length = length
        self.tsp_gen = TSPMaster(num_cities=num_cities)
        super().__init__(length)

    def yield_item(self) -> DataItem:        
        results = None
        while results == None:
            data = self.tsp_gen.generateNetwork()
            results = self.tsp_gen.solve()

        final_matrix = np.zeros((self.num_cities, self.num_cities))
        for edge in results:
            final_matrix[edge[0]._index][edge[1]._index] = 1

        return DataItem(data, target=final_matrix)
    
    def __iter__(self) -> Iterator[dict]:
        for _ in range(self._length):
            next_item = self.yield_item()
            yield {"data": next_item.data}

    def split(
        self, validation_proportion: float = 0.1
    ) -> Tuple[
        'GeneratorDataset', 'GeneratorDataset'
    ]: 
        generator_dataset_type = type(self)
        validation_length = int(self.length * validation_proportion)

        return generator_dataset_type(length = self.length - validation_length, num_cities = self.num_cities), generator_dataset_type(validation_length, num_cities = self.num_cities)