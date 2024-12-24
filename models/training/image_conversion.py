"""Utilities for converting and processing images into numpy arrays."""

from concurrent.futures import ThreadPoolExecutor
from typing import List
import numpy as np
from models.training.model_types import ImageData, ProcessedBatch, ImageArray
import tensorflow as tf
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImageConverter:
    """Handles parallel conversion of PIL images to numpy arrays for batch processing."""

    def __init__(self, num_workers: int = 6):
        self.num_workers = num_workers

    def convert_single_image(self, image):
        """Convert a single image to numpy array with validation"""
        try:
            # Ensure it's a PIL Image
            if not isinstance(image, Image.Image):
                raise ValueError("Input must be a PIL Image")

            # Resize and convert to RGB
            image = image.convert("RGB")
            image = image.resize((96, 96), Image.Resampling.LANCZOS)

            # Convert to numpy array
            image_array = np.array(image)

            # Validate shape
            if image_array.shape != (96, 96, 3):
                raise ValueError(f"Invalid image shape: {image_array.shape}")

            return image_array

        except Exception as e:
            logger.error(f"Error converting image: {str(e)}")
            raise

    def process_batch(self, images):
        """Process a batch of images in parallel"""
        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                processed_images = list(executor.map(self.convert_single_image, images))
            return processed_images

        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            raise ValueError(f"Error processing batch: {str(e)}")
