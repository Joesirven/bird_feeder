"""Type definitions and data structures for training pipeline components."""

from typing import Dict, Union, Protocol, runtime_checkable
from dataclasses import dataclass
from PIL import Image
import numpy as np
import numpy.typing as npt

# Type definitions
ImageArray = npt.NDArray[np.uint8]  # Specifically for uint8 image arrays
Label = int
BatchSize = int


@dataclass
class ImageData:
    """Single image data with its label"""

    image: Union[Image.Image, ImageArray]  # Can be PIL Image or numpy array
    label: Label

    def validate(self) -> bool:
        """Validate the image data"""
        if isinstance(self.image, Image.Image):
            return self.image.mode == "RGB"
        elif isinstance(self.image, np.ndarray):
            return (
                self.image.dtype == np.uint8
                and len(self.image.shape) == 3
                and self.image.shape[2] == 3
            )
        return False


@dataclass
class ProcessedBatch:
    """Batch of processed images and labels"""

    images: npt.NDArray[np.uint8]  # Shape: (batch_size, height, width, 3)
    labels: npt.NDArray[np.int64]

    def validate(self) -> bool:
        """Validate the batch"""
        return (
            self.images.dtype == np.uint8
            and len(self.images.shape) == 4
            and self.images.shape[-1] == 3
            and len(self.labels.shape) == 1
            and self.images.shape[0] == self.labels.shape[0]
        )


@runtime_checkable
class DatasetProtocol(Protocol):
    """Protocol for dataset-like objects"""

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> Dict[str, Union[Image.Image, int]]: ...
