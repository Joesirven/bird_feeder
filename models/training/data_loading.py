from typing import Dict, Optional, Union
from pathlib import Path
import os
from datasets import load_dataset, DatasetDict
from models.training.model_types import DatasetProtocol
from PIL import Image


class DataLoader:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._dataset: Optional[DatasetDict] = None
        self._current_split: Optional[str] = None

    def set_split(self, split: str) -> None:
        """Set the current dataset split"""
        if self._dataset is None:
            self.load_dataset()
        if split not in self._dataset:
            raise ValueError(f"Invalid split: {split}")
        self._current_split = split

    def __len__(self) -> int:
        """Implement protocol requirement"""
        if self._dataset is None or self._current_split is None:
            raise RuntimeError("Dataset not loaded or split not set")
        return len(self._dataset[self._current_split])

    def __getitem__(self, idx: int) -> Dict[str, Union[Image.Image, int]]:
        """Implement protocol requirement"""
        if self._dataset is None or self._current_split is None:
            raise RuntimeError("Dataset not loaded or split not set")
        return self._dataset[self._current_split][idx]

    def load_dataset(self) -> DatasetDict:
        """Load the bird dataset with caching"""
        if self._dataset is None:
            self._dataset = load_dataset(
                "chriamue/bird-species-dataset",
                cache_dir=str(self.cache_dir),
            )
            self._validate_dataset(self._dataset)
        return self._dataset

    def _validate_dataset(self, dataset: DatasetDict) -> None:
        """Validate dataset structure and contents"""
        required_splits = {"train", "validation", "test"}
        if not all(split in dataset for split in required_splits):
            raise ValueError(f"Dataset missing required splits: {required_splits}")

        required_features = {"image", "label"}
        for split in dataset:
            if not all(feat in dataset[split].features for feat in required_features):
                raise ValueError(
                    f"Split {split} missing required features: {required_features}"
                )
