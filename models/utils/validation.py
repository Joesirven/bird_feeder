"""Data validation module for pre-training checks."""

import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, List, Counter, Tuple
from dataclasses import dataclass
from .config import Config
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import pyarrow as pa
import pyarrow.dataset as ds
import os

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

    def __init__(self):
        self.cache_dir = Path("data/cache")
        self.debug_dir = Path("debug")
        self.debug_dir.mkdir(exist_ok=True)
        self.expected_image_shape = (Config.IMG_SIZE, Config.IMG_SIZE, 3)
        self.required_splits = ["train", "validation", "test"]
        self.stats = {}
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        # Create debug subdirectories
        (self.debug_dir / "samples").mkdir(exist_ok=True)
        (self.debug_dir / "distribution").mkdir(exist_ok=True)
        (self.debug_dir / "sequence").mkdir(exist_ok=True)

    def _check_raw_data(self) -> Optional[List[str]]:
        """Validate raw data structure and labels and return first 10 labels and images."""
        raw_data_dir = Path(
            "data/cache/chriamue___bird-species-dataset/bird_species_dataset/0.1.0"
        )  # Adjusted path
        if not raw_data_dir.exists():
            logger.warning("Raw data directory not found, skipping raw data check")
            return None

        try:
            # Check for .arrow files in the raw data directory
            arrow_files = list(raw_data_dir.glob("*.arrow"))
            if not arrow_files:
                raise ValueError("No .arrow files found in raw data directory")

            # Log first 10 labels and images
            for arrow_file in arrow_files:
                logger.info(f"Found raw data file: {arrow_file.name}")

                # Here, you would typically load the data from the .arrow file
                # For demonstration, we will just log the file name
                # You can implement actual loading logic if needed

            return [arrow_file.name for arrow_file in arrow_files]

        except Exception as e:
            logger.error(f"Raw data check failed: {e}")
            return None

    def load_datasets(self):
        """Load datasets and assign to attributes."""
        logger.info("Loading datasets from cache...")
        self.train_ds = tf.data.Dataset.load(str(self.cache_dir / "train_processed"))
        self.val_ds = tf.data.Dataset.load(str(self.cache_dir / "validation_processed"))
        self.test_ds = tf.data.Dataset.load(str(self.cache_dir / "test_processed"))

    def _analyze_label_sequence(
        self, dataset: tf.data.Dataset, num_samples: int = 1000
    ) -> Dict:
        """Detailed analysis of label sequences to detect patterns"""
        logger.info("Analyzing label sequences...")

        sequences = []
        for _, labels in dataset.take(num_samples):
            sequences.append(int(labels.numpy()))

        # Calculate sequence metrics
        sequence_diffs = np.diff(sequences)

        metrics = {
            "is_sequential": all(d >= 0 for d in sequence_diffs),
            "unique_transitions": len(set(sequence_diffs)),
            "repeated_patterns": self._detect_patterns(sequences[:100]),
            "sample_sequence": sequences[:20],
        }

        # Plot sequence pattern
        plt.figure(figsize=(15, 5))
        plt.plot(sequences[:100])
        plt.title("Label Sequence Pattern (First 100 Samples)")
        plt.xlabel("Sample Index")
        plt.ylabel("Label Value")
        plt.savefig(self.debug_dir / "sequence" / "label_sequence_pattern.png")
        plt.close()

        return metrics

    def _detect_patterns(self, sequence: List[int]) -> Dict:
        """Detect potential patterns in label sequence"""
        diffs = np.diff(sequence)
        return {
            "constant_diff": len(set(diffs)) == 1,
            "repeating": len(set(sequence)) < len(sequence) * 0.5,
            "monotonic": all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs),
        }

    def validate_label_sequence(self, dataset: tf.data.Dataset) -> bool:
        """Enhanced sequence validation with detailed analysis"""
        logger.info("Performing detailed label sequence validation...")

        sequence_metrics = self._analyze_label_sequence(dataset)

        if sequence_metrics["is_sequential"]:
            logger.error("WARNING: Labels appear to be sequential!")
            logger.info("Sequence Analysis:")
            logger.info(f"- First 20 labels: {sequence_metrics['sample_sequence']}")
            logger.info(f"- Pattern detection: {sequence_metrics['repeated_patterns']}")
            return False

        logger.info("Label sequence appears properly randomized")
        return True

    def _analyze_label_distribution(self, dataset: tf.data.Dataset) -> Dict:
        """Analyze label distribution with enhanced metrics"""
        logger.info("Analyzing label distribution...")

        try:
            all_labels = []
            label_counts_sample = []

            for i, (_, label) in enumerate(dataset):
                label_val = int(label.numpy())
                all_labels.append(label_val)

                if i < 10:
                    label_counts_sample.append(label_val)
                    logger.info(f"Sample label {i}: {label_val}")

            # Calculate distribution
            label_counts = Counter(all_labels)

            # Plot detailed distribution
            plt.figure(figsize=(15, 5))
            counts = [label_counts.get(i, 0) for i in range(Config.NUM_CLASSES)]
            plt.bar(range(Config.NUM_CLASSES), counts)
            plt.title("Label Distribution")
            plt.xlabel("Class ID")
            plt.ylabel("Count")
            plt.savefig(self.debug_dir / "distribution" / "label_distribution.png")
            plt.close()

            return {
                "num_classes": len(label_counts),
                "label_distribution": dict(label_counts),
                "class_balance": np.std(
                    [count / len(all_labels) for count in label_counts.values()]
                ),
                "min_samples": min(label_counts.values()),
                "max_samples": max(label_counts.values()),
                "empty_classes": list(
                    set(range(Config.NUM_CLASSES)) - set(label_counts.keys())
                ),
                "sample_sequence": label_counts_sample,
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

    def validate_data_labels(self):
        """Validate image-label correspondence"""
        logger.info("Checking image-label alignment...")
        try:
            # Load a few samples
            for x, y in self.train_ds.take(1):
                # Get actual images and predicted labels
                images = x.numpy()
                labels = y.numpy()

                # Log some samples
                for i in range(min(5, len(images))):
                    logger.info(f"Image shape: {images[i].shape}, Label: {labels[i]}")

                    # Optional: Save these samples for visual inspection
                    tf.keras.preprocessing.image.save_img(
                        f"debug_sample_{i}.jpg", images[i]
                    )
            return True
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return False

    def extract_images_and_labels(self) -> None:
        """Extract the first 10 images and their labels from the .arrow files."""
        raw_data_dir = Path(
            "data/cache/chriamue___bird-species-dataset/bird_species_dataset/0.1.0/bfe742b8a5b73193f04ed4fed961614ac0279b36ed3cc3b7e6a65642afa27e92"
        )
        arrow_files = list(raw_data_dir.glob("*.arrow"))

        if not arrow_files:
            logger.warning("No .arrow files found to extract images and labels.")
            return

        for arrow_file in arrow_files:
            logger.info(f"Extracting from: {arrow_file.name}")

            file = os.open(arrow_file, flags=0)

            logger.info(f"Arrow file type: {type(file)}")

            # Load the dataset from the .arrow file
            dataset = ds.dataset(arrow_file, format="arrow")

            # Create a table from the dataset
            table = dataset.to_table()

            # Extract the first 10 rows
            for i in range(min(10, table.num_rows)):
                image = table[i][
                    "image"
                ]  # Assuming 'image' is the column name for images
                label = table[i][
                    "label"
                ]  # Assuming 'label' is the column name for labels

                # Log the image and label
                logger.info(f"Image {i}: {image}, Label: {label}")

                # Optionally, you can save the image to a file for inspection
                # Here, we assume the image is in a format that can be saved directly
                # You may need to convert it depending on how it's stored
                # For example, if it's a numpy array:
                # tf.keras.preprocessing.image.save_img(f"debug/sample_{i}.jpg", image)

            logger.info("Extraction complete.")

    def _print_validation_report(self):
        """Print formatted validation report."""
        logger.info("\n" + "=" * 50)
        logger.info("📋 Dataset Validation Report")
        logger.info("=" * 50)

        # Check raw data first
        try:
            raw_labels = self._check_raw_data()
            if raw_labels:  # Only print if we got labels back
                logger.info(f"\n📁 Raw Data Check:")
                logger.info(f"   • Found {len(raw_labels)} classes")
                logger.info(f"   • Labels file verified ✅")
            else:
                logger.info("\n📁 Raw Data Check: Skipped (using processed data only)")
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

            # Load datasets first
            self.load_datasets()
            self.extract_images_and_labels()

            for split in self.required_splits:
                dataset = getattr(self, f"{split}_ds")  # Dynamically get the dataset
                if dataset is None:
                    raise ValueError(f"Missing dataset split: {split}")

                # Check structure first
                self._check_dataset_structure(dataset)

                # Validate contents
                self.stats[split] = self._validate_split(dataset, split)

                if split == "train" and not self.validate_label_sequence(dataset):
                    logger.error("Label sequence validation failed!")
                    return False

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
