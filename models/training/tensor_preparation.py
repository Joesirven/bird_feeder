"""Utilities for converting processed batches into TensorFlow tensors."""

from typing import Tuple, List
import tensorflow as tf
from .model_types import ProcessedBatch


class TensorPreparator:
    """Handles conversion of processed image batches into TensorFlow-compatible tensors."""

    def __init__(self, img_size: int = 96):
        self.img_size = img_size

    def prepare_batch(self, batch: ProcessedBatch) -> Tuple[tf.Tensor, tf.Tensor]:
        """Convert processed batch to TensorFlow tensors"""
        if not batch.validate():
            raise ValueError("Invalid batch data")

        try:
            # Convert to float32 and normalize
            images = tf.convert_to_tensor(batch.images, dtype=tf.float32)

            # Check if reshaping is needed
            if images.shape[1:3] != (self.img_size, self.img_size):
                images = tf.image.resize(images, (self.img_size, self.img_size))

            # Ensure values are in [0,1] range
            if tf.reduce_max(images) > 1.0:
                images = images / 255.0

            labels = tf.convert_to_tensor(batch.labels, dtype=tf.int64)

            return images, labels

        except Exception as e:
            raise ValueError(f"Error preparing batch: {str(e)}")

    def create_dataset(
        self, batches: List[ProcessedBatch], batch_size: int
    ) -> tf.data.Dataset:
        """Create TensorFlow dataset from processed batches"""
        # Convert all batches to tensors first
        tensor_pairs = [self.prepare_batch(batch) for batch in batches]

        # Combine all images and labels
        all_images = tf.concat([pair[0] for pair in tensor_pairs], axis=0)
        all_labels = tf.concat([pair[1] for pair in tensor_pairs], axis=0)

        # Create dataset from combined tensors
        dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
        return dataset.batch(batch_size)
