import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from models.training.process import main as process_data
from models.training.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles the complete training pipeline for the bird classification model."""

    def __init__(self):
        self.model = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.history = None
        self.checkpoint_dir = Path(Config.CHECKPOINT_PATH)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def setup_model(self):
        """Initialize EfficientNet-B2 with transfer learning architecture"""
        logger.info("Setting up EfficientNet-B2 model...")

        # Load pre-trained base model
        base_model = tf.keras.applications.EfficientNetB2(
            include_top=False,  # Exclude classification head
            weights="imagenet",  # Pre-trained weights
            input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3),
        )

        # Freeze base model for transfer learning
        base_model.trainable = False
        logger.info(f"Base model loaded with {len(base_model.layers)} layers")

        # Create custom model architecture
        model = tf.keras.Sequential(
            [
                # Pre-trained feature extractor
                base_model,
                # Global pooling to reduce feature maps
                tf.keras.layers.GlobalAveragePooling2D(),
                # Regularization
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.BatchNormalization(),
                # Classification head
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(Config.NUM_CLASSES, activation="softmax"),
            ]
        )

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        logger.info("Model setup complete")
        return model

    def load_datasets(self):
        """Load and prepare datasets for training"""
        logger.info("Loading preprocessed datasets...")
        datasets = process_data()

        self.train_ds = datasets["train"]
        self.val_ds = datasets["validation"]
        self.test_ds = datasets["test"]

        logger.info("Datasets loaded successfully")
