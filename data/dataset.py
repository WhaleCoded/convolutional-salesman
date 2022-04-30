from data.graphs import generate_solvable_graph
from typing import Tuple, Iterator
import numpy as np
from typing import Iterator, Tuple
from torch.utils.data import IterableDataset

from typing import Any, Hashable, Sequence, Iterator, Tuple


class DataItem:
    """A single data example for rootflow datasets.

    A container class for data in rootflow datasets, intended to provide a rigid API
    on which the :class:`FunctionalDataset`s can depened. Behaviorally, it is similar
    to a named tuple, since the only available slots are `id`, `data` and `target`.

    Attributes:
        id (:obj:`Hashable`, optional): A unique id for the dataset example.
        data (Any): The data of the dataset example.
        target (:obj:`Any`, optional): The task target(s) for the dataset example.
    """

    __slots__ = ("id", "data", "target")

    # TODO We may want to unpack lists with only a single item for mappings and nested lists as well
    def __init__(self, data: Any, id: Hashable = None, target: Any = None) -> None:
        """Creates a new data item.

        Args:
            id (:obj:`Hashable`, optional): A unique id for the dataset example.
            data (Any): The data of the dataset example.
            target (:obj:`Any`, optional): The task target(s) for the dataset example
        """
        self.data = data
        self.id = id

        if isinstance(target, Sequence) and not isinstance(target, str):
            target_length = len(target)
            if target_length == 0:
                target = None
            elif target_length == 1:
                target = target[0]
        self.target = target

    def __getitem__(self, index: int):
        if index == 0:
            return self.id
        elif index == 1:
            return self.data
        elif index == 2:
            return self.target
        else:
            raise ValueError(f"Invalid index {index} for CollectionDataItem")

    def __iter__(self) -> Iterator[Tuple[Hashable, Any, Any]]:
        """Returns an iterator to support tuple unpacking

        For example:
            >>> data_item = CollectionDataItem([1, 2, 3], id = 'item', target = 0)
            >>> id, data, target = data_item
        """
        return iter((self.id, self.data, self.target))


class GeneratorDataset(IterableDataset):
    def __init__(self, length: int) -> None:
        super().__init__()

        self._length = length

    def __len__(self) -> int:
        return self._length

    def split(
        self, validation_proportion: float = 0.1
    ) -> Tuple["GeneratorDataset", "GeneratorDataset"]:
        generator_dataset_type = type(self)
        validation_length = int(self.length * validation_proportion)

        return generator_dataset_type(
            length=self.length - validation_length
        ), generator_dataset_type(validation_length)

    def __iter__(self) -> Iterator[dict]:
        for _ in range(self._length):
            next_item = self.yield_item()
            yield {"data": next_item.data, "target": next_item.target}

    def yield_item(self) -> DataItem:
        raise NotImplementedError(
            "To create a new GeneratorDataset the method yeild_item must be implemented."
        )


class TspDataset(GeneratorDataset):
    def __init__(self, length: int, num_cities: int = 1000) -> None:
        self.num_cities = num_cities
        self.length = length
        super().__init__(length)

    def yield_item(self) -> DataItem:
        connection_proportion = (np.random.random() * 0.8) + 0.1
        data = generate_solvable_graph(
            num_cities=self.num_cities, connection_proportion=connection_proportion
        )
        return DataItem(data)

    def __iter__(self) -> Iterator[dict]:
        for _ in range(self._length):
            next_item = self.yield_item()
            yield {"data": next_item.data}

    def split(
        self, validation_proportion: float = 0.1
    ) -> Tuple["GeneratorDataset", "GeneratorDataset"]:
        generator_dataset_type = type(self)
        validation_length = int(self.length * validation_proportion)

        return generator_dataset_type(
            length=self.length - validation_length, num_cities=self.num_cities
        ), generator_dataset_type(validation_length, num_cities=self.num_cities)
