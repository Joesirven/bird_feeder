"""Utilities for converting and processing images into numpy arrays."""

from concurrent.futures import ThreadPoolExecutor
from typing import List
import numpy as np
from models.training.model_types import ImageData, ProcessedBatch, ImageArray
import tensorflow as tf


class ImageConverter:
    """Handles parallel conversion of PIL images to numpy arrays for batch processing."""

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers

    def convert_single_image(self, image_data: ImageData) -> ImageArray:
        """Convert single PIL Image to numpy array"""
        if not image_data.validate():
            raise ValueError("Invalid image data")

        if isinstance(image_data.image, np.ndarray):
            return image_data.image

        # Convert PIL Image to numpy array
        array = np.array(image_data.image, dtype=np.uint8)
        if not (array.ndim == 3 and array.shape[2] == 3):
            raise ValueError(f"Invalid image shape: {array.shape}")

        return array

    def process_batch(self, images: List[ImageData], batch_size: int) -> ProcessedBatch:
        """Process a batch of images in parallel"""
        try:
            # Process in smaller chunks to manage memory
            chunk_size = min(batch_size, 32)
            processed_images = []

            # Add debug logging
            shapes = []

            for i in range(0, len(images), chunk_size):
                chunk = images[i : i + chunk_size]
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    chunk_processed = list(
                        executor.map(self.convert_single_image, chunk)
                    )

                    # Debug: Log shapes
                    for img in chunk_processed:
                        shapes.append(img.shape)
                        if (
                            len(processed_images) > 0
                            and img.shape != processed_images[0].shape
                        ):
                            print(
                                f"Shape mismatch! Expected {processed_images[0].shape}, got {img.shape}"
                            )

                    processed_images.extend(chunk_processed)

            # Before stacking, ensure all images are the same size
            target_shape = processed_images[0].shape
            normalized_images = []

            for img in processed_images:
                if img.shape != target_shape:
                    # Resize to match target shape
                    resized_img = tf.image.resize(
                        img, (target_shape[0], target_shape[1])
                    )
                    normalized_images.append(np.array(resized_img, dtype=np.uint8))
                else:
                    normalized_images.append(img)

            # Stack images and create labels array
            batch_images = np.stack(normalized_images)
            batch_labels = np.array([img.label for img in images], dtype=np.int64)

            batch = ProcessedBatch(images=batch_images, labels=batch_labels)
            if not batch.validate():
                raise ValueError("Invalid processed batch")

            return batch

        except Exception as e:
            raise ValueError(f"Error processing batch: {str(e)}")
