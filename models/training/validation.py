"""Data validation module for pre-training checks."""

import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from .config import Config
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("validation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Statistics for a dataset split."""

    num_samples: int
    image_shape: tuple
    label_range: tuple
    dtype_images: str
    dtype_labels: str
    memory_usage_mb: float


class DataValidator:
    """Validates processed TensorFlow datasets before training."""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.required_splits = ["train", "validation", "test"]
        self.expected_image_shape = (Config.IMG_SIZE, Config.IMG_SIZE, 3)
        self.stats: Dict[str, DatasetStats] = {}

    def validate_all(self) -> bool:
        """Run all validation checks and return success status."""
        try:
            logger.info("📊 Starting dataset validation...")

            # Load and validate each split
            for split in self.required_splits:
                dataset_path = self.cache_dir / f"{split}_processed"
                if not dataset_path.exists():
                    raise ValueError(f"Missing dataset split: {split}")

                # Load dataset
                dataset = tf.data.Dataset.load(str(dataset_path))
                self.stats[split] = self._validate_split(dataset, split)

            # Print validation report
            self._print_validation_report()
            return True

        except Exception as e:
            logger.error(f"❌ Validation failed: {str(e)}")
            return False

    def _validate_split(self, dataset: tf.data.Dataset, split: str) -> DatasetStats:
        """Validate a single dataset split."""
        logger.info(f"Validating {split} split...")

        # Get first batch for inspection
        for images, labels in dataset.take(1):
            image_shape = images.shape[1:]  # Remove batch dimension
            if image_shape != self.expected_image_shape:
                raise ValueError(f"Invalid image shape in {split}: {image_shape}")

        # Calculate statistics
        num_samples = len(list(dataset))
        sample_images, sample_labels = next(iter(dataset))

        memory_usage = (
            sample_images.numpy().nbytes * num_samples
            + sample_labels.numpy().nbytes * num_samples
        ) / (
            1024 * 1024
        )  # Convert to MB

        return DatasetStats(
            num_samples=num_samples,
            image_shape=image_shape,
            label_range=(
                int(tf.reduce_min(sample_labels)),
                int(tf.reduce_max(sample_labels)),
            ),
            dtype_images=str(sample_images.dtype),
            dtype_labels=str(sample_labels.dtype),
            memory_usage_mb=memory_usage,
        )

    def _print_validation_report(self):
        """Print formatted validation report."""
        logger.info("\n" + "=" * 50)
        logger.info("📋 Dataset Validation Report")
        logger.info("=" * 50)

        for split, stats in self.stats.items():
            logger.info(f"\n📦 {split.upper()} Split:")
            logger.info(f"   • Samples: {stats.num_samples:,}")
            logger.info(f"   • Image Shape: {stats.image_shape}")
            logger.info(f"   • Label Range: {stats.label_range}")
            logger.info(f"   • Image dtype: {stats.dtype_images}")
            logger.info(f"   • Label dtype: {stats.dtype_labels}")
            logger.info(f"   • Memory Usage: {stats.memory_usage_mb:.2f} MB")

        logger.info("\n✅ All validation checks passed!")
        logger.info("=" * 50)


if __name__ == "__main__":
    validator = DataValidator()
    if validator.validate_all():
        print("\n✅ Datasets validated successfully!")
    else:
        print("\n❌ Dataset validation failed!")
