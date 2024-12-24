"""Main processing pipeline for loading, converting, and preparing."""

from pathlib import Path
from .data_loading import DataLoader
from .image_conversion import ImageConverter
from .tensor_preparation import TensorPreparator
from .model_types import ImageData
from tqdm import tqdm
import tensorflow as tf
import logging
from .config import Config
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("preprocessing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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
                processed_batch = converter.process_batch(batch_images)

                # Extend our lists with the processed data
                processed_images.extend(processed_batch)
                processed_labels.extend(batch_labels)

                pbar.update(len(batch_indices))

        # Convert to TensorFlow dataset
        logger.info(f"Creating TensorFlow dataset for {split}")
        images_tensor = tf.convert_to_tensor(processed_images, dtype=tf.float32)
        labels_tensor = tf.convert_to_tensor(processed_labels, dtype=tf.int64)

        # Create and save dataset
        tf_dataset = tf.data.Dataset.from_tensor_slices((images_tensor, labels_tensor))
        save_path = cache_dir / f"{split}_processed"
        tf.data.Dataset.save(tf_dataset, str(save_path))

        logger.info(f"Dataset shapes for {split}:")
        logger.info(f"Images shape: {images_tensor.shape}")
        logger.info(f"Labels shape: {labels_tensor.shape}")

        processed_datasets[split] = tf_dataset

    return processed_datasets


if __name__ == "__main__":
    try:
        datasets = main()
        print("\nDataset processing completed successfully!")
        for split, ds in datasets.items():
            print(f"{split} dataset size: {len(list(ds))} batches")
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise
