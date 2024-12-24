"""Data validation module for pre-training checks."""

import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, List, Counter
from dataclasses import dataclass
from .config import Config
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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
    label_distribution: Dict[int, int]
    num_classes: int
    class_balance: float


class DataValidator:
    """Validates processed TensorFlow datasets before training."""

    def __init__(self, cache_dir: str = "data/cache", data_dir: str = "data"):
        self.cache_dir = Path(cache_dir)
        self.data_dir = Path(data_dir)
        self.required_splits = ["train", "validation", "test"]
        self.expected_image_shape = (Config.IMG_SIZE, Config.IMG_SIZE, 3)
        self.expected_num_classes = Config.NUM_CLASSES
        self.stats: Dict[str, DatasetStats] = {}

    def _check_raw_data(self) -> None:
        """Verify raw data and labels exist and match."""
        logger.info("Checking raw data integrity...")

        # Check data directory structure
        data_path = self.data_dir / "birds"
        if not data_path.exists():
            raise ValueError(f"Data directory not found: {data_path}")

        # Check label files
        label_file = self.data_dir / "birds" / "labels.txt"
        if not label_file.exists():
            raise ValueError(f"Labels file not found: {label_file}")

        # Read and validate labels
        with open(label_file, "r") as f:
            labels = [line.strip() for line in f.readlines()]

        if len(labels) != Config.NUM_CLASSES:
            raise ValueError(
                f"Expected {Config.NUM_CLASSES} classes, found {len(labels)}"
            )

        logger.info(f"Found {len(labels)} classes in raw data")
        return labels

    def _analyze_label_distribution(self, dataset: tf.data.Dataset) -> Dict[str, float]:
        """Analyze the distribution of labels in the dataset."""
        logger.info("Analyzing label distribution...")

        try:
            # Collect all labels with detailed logging
            all_labels = []
            label_counts_sample = []  # Track first few labels

            for i, (_, label) in enumerate(dataset):
                label_val = int(label.numpy())
                all_labels.append(label_val)

                # Log first 10 labels for debugging
                if i < 10:
                    label_counts_sample.append(label_val)
                    logger.info(f"Sample label {i}: {label_val}")

            # Calculate distribution
            label_counts = Counter(all_labels)

            # Log detailed distribution stats
            logger.info("\nLabel Distribution Summary:")
            logger.info(f"Total unique labels: {len(label_counts)}")
            logger.info(f"Min samples per class: {min(label_counts.values())}")
            logger.info(f"Max samples per class: {max(label_counts.values())}")

            # Find empty classes
            missing_classes = set(range(Config.NUM_CLASSES)) - set(label_counts.keys())
            if missing_classes:
                logger.warning(f"Missing classes: {sorted(missing_classes)}")

            # Plot detailed distribution
            plt.figure(figsize=(15, 5))
            counts = [label_counts.get(i, 0) for i in range(Config.NUM_CLASSES)]
            plt.bar(range(Config.NUM_CLASSES), counts)
            plt.title("Label Distribution")
            plt.xlabel("Class ID")
            plt.ylabel("Count")
            plt.savefig(f"label_distribution_detailed.png")
            plt.close()

            return {
                "num_classes": len(label_counts),
                "label_distribution": dict(label_counts),
                "class_balance": np.std(
                    [count / len(all_labels) for count in label_counts.values()]
                ),
                "min_samples": min(label_counts.values()),
                "max_samples": max(label_counts.values()),
                "empty_classes": list(missing_classes),
                "sample_sequence": label_counts_sample[:10],  # Store first 10 labels
            }

        except Exception as e:
            logger.error(f"Error analyzing label distribution: {str(e)}")
            raise

    def _check_dataset_structure(self, dataset: tf.data.Dataset) -> None:
        """Verify the dataset has the expected structure."""

        # Check element spec
        spec = dataset.element_spec
        logger.info(f"Dataset element spec: {spec}")

        if not isinstance(spec, tuple) or len(spec) != 2:
            raise ValueError(
                f"Dataset should contain (image, label) pairs, got: {spec}"
            )

        image_spec, label_spec = spec

        # Verify image spec
        if image_spec.dtype != tf.float32:
            raise ValueError(f"Expected float32 images, got {image_spec.dtype}")

        # Verify label spec
        if label_spec.dtype != tf.int64:
            raise ValueError(f"Expected int64 labels, got {label_spec.dtype}")

    def _validate_split(self, dataset: tf.data.Dataset, split: str) -> DatasetStats:
        """Validate a single dataset split."""
        logger.info(f"Validating {split} split...")

        try:
            # Debug dataset structure
            logger.info("Examining dataset structure...")

            # Get first batch and print shapes
            for images, labels in dataset.take(1):
                logger.info(
                    f"Raw tensor shapes - Images: {images.shape}, Labels: {labels.shape}"
                )

                # Handle both batched and unbatched data
                if len(images.shape) == 4:  # Batched data
                    image_shape = images.shape[1:]
                elif len(images.shape) == 3:  # Single image
                    image_shape = images.shape
                else:
                    raise ValueError(f"Unexpected image tensor shape: {images.shape}")

                logger.info(f"Detected image shape: {image_shape}")

                if image_shape != self.expected_image_shape:
                    raise ValueError(
                        f"Invalid image shape in {split}:\n"
                        f"Expected {self.expected_image_shape}, got {image_shape}"
                    )

            # Calculate statistics
            num_samples = len(list(dataset))
            sample_images, sample_labels = next(iter(dataset))

            # Get label distribution stats
            dist_stats = self._analyze_label_distribution(dataset)

            memory_usage = (
                sample_images.numpy().nbytes * num_samples
                + sample_labels.numpy().nbytes * num_samples
            ) / (
                1024 * 1024
            )  # Convert to MB

            stats = DatasetStats(
                num_samples=num_samples,
                image_shape=image_shape,
                label_range=(
                    min(dist_stats["label_distribution"].keys()),
                    max(dist_stats["label_distribution"].keys()),
                ),
                dtype_images=str(sample_images.dtype),
                dtype_labels=str(sample_labels.dtype),
                memory_usage_mb=memory_usage,
                label_distribution=dist_stats["label_distribution"],
                num_classes=dist_stats["num_classes"],
                class_balance=dist_stats["class_balance"],
            )

            logger.info(f"Split statistics calculated successfully")
            return stats

        except Exception as e:
            logger.error(f"Error validating {split} split: {str(e)}")
            raise

    def _print_validation_report(self):
        """Print formatted validation report."""
        logger.info("\n" + "=" * 50)
        logger.info("📋 Dataset Validation Report")
        logger.info("=" * 50)

        # Check raw data first
        try:
            raw_labels = self._check_raw_data()
            logger.info(f"\n📁 Raw Data Check:")
            logger.info(f"   • Found {len(raw_labels)} classes")
            logger.info(f"   • Labels file verified ✅")
        except Exception as e:
            logger.error(f"Raw data check failed: {str(e)}")

        for split, stats in self.stats.items():
            logger.info(f"\n📦 {split.upper()} Split:")
            logger.info(f"   • Samples: {stats.num_samples:,}")
            logger.info(f"   • Image Shape: {stats.image_shape}")
            logger.info(f"   • Label Range: {stats.label_range}")
            logger.info(f"   • Classes Found: {stats.num_classes}")
            logger.info(f"   • Class Balance: {stats.class_balance:.2f}")
            logger.info(f"   • Image dtype: {stats.dtype_images}")
            logger.info(f"   • Label dtype: {stats.dtype_labels}")
            logger.info(f"   • Memory Usage: {stats.memory_usage_mb:.2f} MB")

            if stats.num_classes < Config.NUM_CLASSES:
                logger.warning(
                    f"⚠️  Missing {Config.NUM_CLASSES - stats.num_classes} classes!"
                )

        logger.info("\n✅ All validation checks passed!")
        logger.info("=" * 50)

    def validate_all(self) -> bool:
        """Run all validation checks and return success status."""
        try:
            logger.info("📊 Starting dataset validation...")

            for split in self.required_splits:
                dataset_path = self.cache_dir / f"{split}_processed"
                if not dataset_path.exists():
                    raise ValueError(f"Missing dataset split: {split}")

                # Load dataset
                dataset = tf.data.Dataset.load(str(dataset_path))

                # Check structure first
                self._check_dataset_structure(dataset)

                # Then validate contents
                self.stats[split] = self._validate_split(dataset, split)

            # Print validation report
            self._print_validation_report()
            return True

        except Exception as e:
            logger.error(f"❌ Validation failed: {str(e)}")
            return False


if __name__ == "__main__":
    validator = DataValidator()
    if validator.validate_all():
        print("\n✅ Datasets validated successfully!")
    else:
        print("\n❌ Dataset validation failed!")
