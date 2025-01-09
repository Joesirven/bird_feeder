"""Main processing pipeline for loading, converting, and preparing."""

from pathlib import Path
from ..training.data_loading import DataLoader
from ..training.image_conversion import ImageConverter
from ..training.tensor_preparation import TensorPreparator
from ..training.model_types import ImageData
from tqdm import tqdm
import tensorflow as tf
import logging
from .config import Config
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("preprocessing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def create_dataset_in_chunks(data, labels, chunk_size=1000):
    """Create dataset in chunks to avoid memory issues"""
    datasets = []
    for i in range(0, len(data), chunk_size):
        chunk_data = data[i : i + chunk_size]
        chunk_labels = labels[i : i + chunk_size]
        with tf.device("/CPU:0"):
            dataset = tf.data.Dataset.from_tensor_slices(
                (
                    tf.convert_to_tensor(chunk_data, dtype=tf.float32),
                    tf.convert_to_tensor(chunk_labels, dtype=tf.int64),
                )
            )
        datasets.append(dataset)
    return datasets[0] if len(datasets) == 1 else datasets[0].concatenate(*datasets[1:])


def main():
    """Execute the complete data processing pipeline for model training."""
    logger.info("Starting preprocessing pipeline...")

    # Initialize components
    cache_dir = Path("data/cache")
    cache_dir.mkdir(exist_ok=True, parents=True)

    loader = DataLoader(cache_dir=cache_dir)
    converter = ImageConverter(num_workers=6)

    # Load dataset
    dataset = loader.load_dataset()
    processed_datasets = {}

    # Process in batches
    batch_size = Config.BATCH_SIZE
    for split in dataset:
        logger.info(f"Processing {split} split...")

        # Convert images to numpy arrays
        processed_images = []
        processed_labels = []
        total_samples = len(dataset[split])

        with tqdm(total=total_samples, desc=f"Processing {split}") as pbar:
            for i in range(0, total_samples, batch_size):
                # Get batch using select
                batch_indices = range(i, min(i + batch_size, total_samples))
                batch_data = dataset[split].select(batch_indices)

                # Convert batch
                batch_images = [item["image"] for item in batch_data]
                batch_labels = [item["label"] for item in batch_data]

                # Process images in parallel
                try:
                    processed_batch = converter.process_batch(batch_images)
                except Exception as e:
                    logger.error(f"Error processing batch {i} for {split}: {str(e)}")
                    continue  # Skip this batch and continue with the next

                # Extend our lists with the processed data
                processed_images.extend(processed_batch)
                processed_labels.extend(batch_labels)

                pbar.update(len(batch_indices))

        # Shuffle the processed data
        logger.info(f"Shuffling {split} dataset...")
        idx = np.random.RandomState(42).permutation(len(processed_images))
        processed_images = np.array(processed_images)[idx]
        processed_labels = np.array(processed_labels)[idx]

        # Convert to TensorFlow dataset using chunks
        logger.info(f"Creating TensorFlow dataset for {split}")
        try:
            # Process tensors in chunks on CPU
            logger.info("Converting images to tensor (in chunks)...")
            images_tensor = create_dataset_in_chunks(processed_images, processed_labels)

            logger.info("Converting labels to tensor (in chunks)...")
            labels_tensor = create_dataset_in_chunks(processed_labels, processed_labels)

            # Create and save dataset
            tf_dataset = tf.data.Dataset.from_tensor_slices(
                (images_tensor, labels_tensor)
            )
            save_path = cache_dir / f"{split}_processed"

            logger.info(f"Saving dataset to {save_path}")
            tf.data.Dataset.save(tf_dataset, str(save_path))

            logger.info(f"Dataset shapes for {split}:")
            logger.info(f"Images shape: {images_tensor.shape}")
            logger.info(f"Labels shape: {labels_tensor.shape}")

            processed_datasets[split] = tf_dataset

        except Exception as e:
            logger.error(f"Error creating dataset for {split}: {str(e)}")
            raise

    return processed_datasets


def process_data(data_dir: Path):
    """Process and shuffle data properly"""
    logger.info("Starting data reprocessing with proper shuffling...")

    # Load all image paths and labels
    image_paths = []
    labels = []
    for class_idx, class_dir in enumerate(sorted(data_dir.glob("*/"))):
        for img_path in class_dir.glob("*.jpg"):
            image_paths.append(img_path)
            labels.append(class_idx)

    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # Shuffle with fixed seed
    idx = np.random.RandomState(42).permutation(len(image_paths))
    image_paths = image_paths[idx]
    labels = labels[idx]

    logger.info(f"Total images: {len(image_paths)}")
    logger.info(f"Unique labels: {len(np.unique(labels))}")

    return image_paths, labels


if __name__ == "__main__":
    try:
        datasets = main()
        print("\nDataset processing completed successfully!")
        for split, ds in datasets.items():
            print(f"{split} dataset size: {len(list(ds))} batches")
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise
