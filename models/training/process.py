"""Main processing pipeline for loading, converting, and preparing."""

from pathlib import Path
from .data_loading import DataLoader
from .image_conversion import ImageConverter
from .tensor_preparation import TensorPreparator
from .model_types import ImageData
from tqdm import tqdm


def main():
    """Execute the complete data processing pipeline for model training."""

    # Initialize components
    loader = DataLoader(cache_dir=Path("data/cache"))
    converter = ImageConverter(num_workers=6)
    preparator = TensorPreparator(img_size=96)

    # Load dataset
    dataset = loader.load_dataset()
    processed_datasets = {}

    # Process in batches
    batch_size = 32
    for split in dataset:
        print(f"Processing {split} split...")

        # Convert images to numpy arrays
        processed_batches = []
        total_samples = len(dataset[split])

        # Create progress bar
        with tqdm(total=total_samples, desc=f"Processing {split}") as pbar:
            # Use select() method instead of direct slicing
            for i in range(0, total_samples, batch_size):
                # Get batch using select
                batch_indices = range(i, min(i + batch_size, total_samples))
                batch_data = dataset[split].select(batch_indices)

                # Convert to ImageData objects
                image_data = [
                    ImageData(image=item["image"], label=item["label"])
                    for item in batch_data
                ]

                # Process batch
                processed_batch = converter.process_batch(image_data, batch_size)
                processed_batches.append(processed_batch)
                pbar.update(len(batch_data))

        # Create TensorFlow dataset
        processed_datasets[split] = preparator.create_dataset(
            processed_batches, batch_size
        )

    return processed_datasets

    return processed_datasets


if __name__ == "__main__":
    try:
        datasets = main()
        print("\nDataset processing completed successfully!")
        for split, ds in datasets.items():
            print(f"{split} dataset size: {len(list(ds))} batches")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
