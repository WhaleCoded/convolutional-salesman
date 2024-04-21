from dataclasses import dataclass
import os
import torch
import json
from typing import List, Optional, Dict

from model import ConvolutionalSalesmanNet

CHECKPOINT_MODEL_NAME = "model.pt"
CHECKPOINT_OPTIMIZER_NAME = "optimizer.pt"
CHECKPOINT_SCHEDULER_NAME = "scheduler.pt"
CHECKPOINT_VALIDATION_UUIDS_NAME = "validation_uuids.json"
CHECKPOINT_META_DATA_NAME = "meta_data.json"
METRIC_NAME = "metrics.json"

SUGGESTED_EPOCHS = 50
SUGGESTED_TRAIN_EXAMPLES = 225_000


class MetaData:
    num_batches_trained: int

    def __init__(self, num_batches_trained: int = 0) -> None:
        self.num_batches_trained = num_batches_trained

    def store_as_json(self, storage_path: os.PathLike) -> None:
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        meta_data_path = os.path.join(storage_path, CHECKPOINT_META_DATA_NAME)

        as_dict = {
            "num_batches_trained": self.num_batches_trained,
        }

        with open(meta_data_path, "w+") as f:
            json.dump(as_dict, f)

    @classmethod
    def load_from_json(cls, storage_path: os.PathLike) -> "MetaData":
        meta_data_path = os.path.join(storage_path, CHECKPOINT_META_DATA_NAME)

        if not os.path.exists(meta_data_path):
            return cls()

        with open(meta_data_path, "r") as f:
            meta_data_dict = json.load(f)

        return cls(**meta_data_dict)


class Metrics:
    training_loss: List[float]
    validation_loss: List[float]
    path_construction_metrics: List[float]
    learning_rate: List[float]

    def __init__(
        self,
        training_loss: List[float] = [],
        validation_loss: List[float] = [],
        path_construction_metrics: List[float] = [],
        learning_rate: List[float] = [],
    ) -> None:
        self.training_loss = training_loss
        self.validation_loss = validation_loss
        self.path_construction_metrics = path_construction_metrics
        self.learning_rate = learning_rate

    def store_as_json(self, storage_path: os.PathLike) -> None:
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        metrics_path = os.path.join(storage_path, METRIC_NAME)

        as_dict = {
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "path_construction_metrics": self.path_construction_metrics,
            "learning_rate": self.learning_rate,
        }

        with open(metrics_path, "w+") as f:
            json.dump(as_dict, f)

    @classmethod
    def load_from_json(cls, storage_path: os.PathLike) -> "Metrics":
        metrics_path = os.path.join(storage_path, METRIC_NAME)

        if not os.path.exists(metrics_path):
            return cls()

        with open(metrics_path, "r") as f:
            metrics_dict = json.load(f)

        return cls(**metrics_dict)


@dataclass
class Checkpoint:
    model_state_dict: Dict[str, torch.Tensor]
    optimizer_state_dict: Dict[str, torch.Tensor]
    lr_scheduler_state_dict: Dict[str, torch.Tensor]
    metrics: Metrics
    metadata: MetaData


class Config:
    data_path: os.PathLike
    checkpoint_path: os.PathLike
    bssf_path: os.PathLike
    device: str
    batches_per_checkpoint: int
    validation_tot_size: int
    train_problem_size_cutoff: int
    test_problem_size_cutoff: int
    min_lr: float
    max_lr: float
    num_path_variations_per_example: int
    tot_train_batches: int

    def __init__(
        self,
        data_path: os.PathLike = os.getenv("DATA_PATH", "./data"),
        checkpoint_path: os.PathLike = os.getenv("CHECKPOINT_PATH", "./checkpoint"),
        bssf_path: os.PathLike = os.getenv("BSSF_PATH", "./bssf"),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batches_per_checkpoint: int = int(os.getenv("BATCHES_PER_CHECKPOINT", 5000)),
        validation_tot_size: int = int(os.getenv("VALIDATION_TOTAL_SIZE", 1000)),
        train_problem_size_cutoff: int = int(
            os.getenv("TRAIN_PROBLEM_SIZE_CUTOFF", 100)
        ),
        test_problem_size_cutoff: int = int(os.getenv("TEST_PROBLEM_SIZE_CUTOFF", 201)),
        min_lr: float = float(os.getenv("MIN_LR", 0.0001)),
        max_lr: float = float(os.getenv("MAX_LR", 0.1)),
        num_path_variations_per_example: int = int(
            os.getenv("NUM_PATH_VARIATIONS_PER_EXAMPLE", 32)
        ),
        tot_train_batches: int = int(
            os.getenv(
                "TOTAL_TRAINING_BATCHES", SUGGESTED_EPOCHS * SUGGESTED_TRAIN_EXAMPLES
            )
        ),
    ) -> None:
        self.data_path = data_path
        self.checkpoint_path = checkpoint_path
        self.bssf_path = bssf_path
        self.device = device
        self.batches_per_checkpoint = batches_per_checkpoint
        self.validation_tot_size = validation_tot_size
        self.train_problem_size_cutoff = train_problem_size_cutoff
        self.test_problem_size_cutoff = test_problem_size_cutoff
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_path_variations_per_example = num_path_variations_per_example
        self.tot_train_batches = tot_train_batches

    def store_as_json(self, config_path: os.PathLike) -> None:
        import json

        as_dict = {
            "data_path": self.data_path,
            "checkpoint_path": self.checkpoint_path,
            "bssf_path": self.bssf_path,
            "device": self.device,
            "batches_per_checkpoint": self.batches_per_checkpoint,
            "validation_tot_size": self.validation_tot_size,
            "train_problem_size_cutoff": self.train_problem_size_cutoff,
            "test_problem_size_cutoff": self.test_problem_size_cutoff,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "num_path_variations_per_example": self.num_path_variations_per_example,
            "tot_train_batches": self.tot_train_batches,
        }

        with open(config_path, "w") as f:
            json.dump(as_dict, f)

    @classmethod
    def from_json(cls, config_path: os.PathLike) -> "Config":
        import json

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def get_validation_uuids(self) -> Optional[List[str]]:
        vald_uuids_path = os.path.join(
            self.checkpoint_path, CHECKPOINT_VALIDATION_UUIDS_NAME
        )
        if not os.path.exists(vald_uuids_path):
            return None

        with open(vald_uuids_path, "r") as f:
            json_data = json.load(f)

        return json_data["uuids"]

    def store_validation_uuids(self, uuids: List[str]) -> None:
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        vald_uuids_path = os.path.join(
            self.checkpoint_path, CHECKPOINT_VALIDATION_UUIDS_NAME
        )

        with open(vald_uuids_path, "w+") as f:
            json.dump({"uuids": uuids}, f)

    def get_current_metrics(self, bssf: bool = False) -> Metrics:
        if bssf:
            checkpoint_metric_path = os.path.join(self.bssf_path, METRIC_NAME)
        else:
            checkpoint_metric_path = os.path.join(self.checkpoint_path, METRIC_NAME)

        if not os.path.exists(checkpoint_metric_path):
            return Metrics()

        return Metrics.load_from_json(self.checkpoint_path)

    def store_metrics(self, metrics: Metrics, bssf: bool = False) -> None:
        if bssf:
            metrics.store_as_json(self.bssf_path)
        else:
            metrics.store_as_json(self.checkpoint_path)

    def store_new_bssf(self, model: torch.nn.Module, metrics: Metrics) -> None:
        if not os.path.exists(self.bssf_path):
            os.makedirs(self.bssf_path)

        model_path = os.path.join(self.bssf_path, CHECKPOINT_MODEL_NAME)
        torch.save(model.state_dict(), model_path)

        metrics.store_as_json(self.bssf_path)

    def get_bssf_metric(self) -> Optional[float]:
        metrics = self.get_current_metrics(bssf=True)
        if len(metrics.path_construction_metrics) == 0:
            return None

        return min(metrics.path_construction_metrics)

    def get_bssf_validation_loss(self) -> Optional[float]:
        metrics = self.get_current_metrics(bssf=True)
        if len(metrics.validation_loss) == 0:
            return None

        return metrics.validation_loss[-1]

    def get_bssf_model_state_dict(self) -> Optional[torch.nn.Module]:
        model_path = os.path.join(self.bssf_path, CHECKPOINT_MODEL_NAME)
        if not os.path.exists(model_path):
            return None

        return torch.load(model_path)

    def store_new_checkpoint(
        self,
        checkpoint: Checkpoint,
    ):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        model_path = os.path.join(self.checkpoint_path, CHECKPOINT_MODEL_NAME)
        torch.save(checkpoint.model_state_dict, model_path)

        optimizer_path = os.path.join(self.checkpoint_path, CHECKPOINT_OPTIMIZER_NAME)
        torch.save(checkpoint.optimizer_state_dict, optimizer_path)

        scheduler_path = os.path.join(self.checkpoint_path, CHECKPOINT_SCHEDULER_NAME)
        torch.save(checkpoint.lr_scheduler_state_dict, scheduler_path)

        checkpoint.metrics.store_as_json(self.checkpoint_path)

        checkpoint.metadata.store_as_json(self.checkpoint_path)

    def get_curr_checkpoint(self) -> Optional[Checkpoint]:
        model_path = os.path.join(self.checkpoint_path, CHECKPOINT_MODEL_NAME)
        if not os.path.exists(model_path):
            return None

        model_state_dict = torch.load(model_path)

        optimizer_path = os.path.join(self.checkpoint_path, CHECKPOINT_OPTIMIZER_NAME)
        optimizer_state_dict = torch.load(optimizer_path)

        scheduler_path = os.path.join(self.checkpoint_path, CHECKPOINT_SCHEDULER_NAME)
        scheduler_state_dict = torch.load(scheduler_path)

        metadata = self.get_curr_metadata()

        metrics = self.get_current_metrics()

        return Checkpoint(
            model_state_dict=model_state_dict,
            optimizer_state_dict=optimizer_state_dict,
            lr_scheduler_state_dict=scheduler_state_dict,
            metrics=metrics,
            metadata=metadata,
        )

    def get_curr_metadata(self) -> MetaData:
        metadata_path = os.path.join(self.checkpoint_path, CHECKPOINT_META_DATA_NAME)
        if not os.path.exists(metadata_path):
            return MetaData()

        return MetaData.load_from_json(self.checkpoint_path)
