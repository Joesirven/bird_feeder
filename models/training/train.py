import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from models.training.process import main as process_data
from models.training.config import Config
from .validation import DataValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Configure Metal GPU for M1
try:
    # Enable Metal GPU acceleration
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        # Allow memory growth
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # Set memory limit (in MB)
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)],
        )
        logger.info(f"Metal GPU enabled: {physical_devices}")

        # Enable mixed precision for better performance
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision enabled")
    else:
        logger.warning("No Metal GPU found")
except Exception as e:
    logger.error(f"Error configuring Metal GPU: {e}")


class ModelTrainer:
    def __init__(self):
        # Validate dataset
        validator = DataValidator()
        if not validator.validate_all():
            raise ValueError("Dataset validation failed. Exiting.")

        self.model = None
        self.train_ds = validator.train_ds
        self.val_ds = validator.val_ds
        self.test_ds = validator.test_ds
        self.history = None
        self.checkpoint_dir = Path(Config.CHECKPOINT_PATH)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.class_counts = None

        # Use appropriate strategy for Metal GPU
        if len(tf.config.list_physical_devices("GPU")) > 0:
            self.strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
            logger.info("Using GPU strategy")
        else:
            self.strategy = tf.distribute.get_strategy()
            logger.warning("Using default (CPU) strategy")

    def monitor_gpu(self):
        """Monitor GPU memory usage"""
        try:
            gpu_devices = tf.config.list_physical_devices("GPU")
            if gpu_devices:
                memory_info = tf.config.experimental.get_memory_info("GPU:0")
                logger.info(
                    f"GPU Memory: {memory_info['current'] / 1024**2:.2f}MB used"
                )
        except Exception as e:
            logger.warning(f"Could not get GPU memory info: {e}")

    def load_datasets(self):
        """Load datasets and compute class distribution"""
        logger.info("Loading cached preprocessed datasets...")
        try:
            cache_dir = Path("data/cache")

            def load_and_prepare(split_name, shuffle=True):
                ds = tf.data.Dataset.load(str(cache_dir / f"{split_name}_processed"))

                if shuffle:
                    ds = ds.shuffle(
                        buffer_size=Config.SHUFFLE_BUFFER,
                        seed=42,
                        reshuffle_each_iteration=True,
                    )

                return ds.map(
                    lambda x, y: (
                        tf.ensure_shape(tf.cast(x, tf.float32) / 255.0, [96, 96, 3]),
                        y,
                    ),
                    num_parallel_calls=tf.data.AUTOTUNE,
                )

            # Load datasets
            train_raw = load_and_prepare(
                "train", shuffle=False
            )  # No shuffle for counting
            self.train_ds = train_raw.batch(Config.BATCH_SIZE).prefetch(
                tf.data.AUTOTUNE
            )
            self.val_ds = (
                load_and_prepare("validation", shuffle=False)
                .batch(Config.BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE)
            )
            self.test_ds = (
                load_and_prepare("test", shuffle=False)
                .batch(Config.BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE)
            )

            # Count class distribution
            logger.info("Calculating class distribution...")
            self.class_counts = {}
            for _, labels in train_raw:  # Use unbatched dataset
                label = int(labels.numpy())
                self.class_counts[label] = self.class_counts.get(label, 0) + 1

            logger.info(f"Found {len(self.class_counts)} classes")
            logger.info(f"Min samples per class: {min(self.class_counts.values())}")
            logger.info(f"Max samples per class: {max(self.class_counts.values())}")

            # Verify shapes
            for x, y in self.train_ds.take(1):
                logger.info(f"Input shape: {x.shape}, Label shape: {y.shape}")

            return True

        except Exception as e:
            logger.error(f"Failed to load datasets: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Failed to load cached datasets: {str(e)}")
            return False

    def setup_model(self):
        """Initialize EfficientNet-B2 model with GPU optimizations"""
        logger.info("Setting up EfficientNet-B2 model...")

        try:
            with self.strategy.scope():
                base_model = tf.keras.applications.EfficientNetB2(
                    include_top=False,
                    weights="imagenet",
                    input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3),
                )

                # Partial unfreezing
                base_model.trainable = True
                for layer in base_model.layers[:-30]:
                    layer.trainable = False

                # Build model with GPU-optimized layers
                inputs = tf.keras.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3))
                x = base_model(inputs)
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dropout(0.6)(x)
                x = tf.keras.layers.Dense(1024, activation="relu")(x)
                x = tf.keras.layers.Dropout(0.6)(x)
                outputs = tf.keras.layers.Dense(
                    Config.NUM_CLASSES,
                    activation="softmax",
                    dtype="float32",  # Ensure final output is float32 for mixed precision
                )(x)

                self.model = tf.keras.Model(inputs, outputs)

                # Compile with GPU-optimized settings
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                    jit_compile=True,  # Enable XLA optimization
                )

                self.model.summary(print_fn=logger.info)
                return True

        except Exception as e:
            logger.error(f"Failed to setup model: {str(e)}")
            return False

    def train(self):
        """Execute training with GPU monitoring"""
        try:
            # Setup callbacks with GPU monitoring
            callbacks = [
                # Save best model
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(self.checkpoint_dir / "best_model.keras"),
                    save_best_only=True,
                    monitor="val_accuracy",
                ),
                # Stop if not improving
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=3, restore_best_weights=True
                ),
                # Monitor training with GPU info
                tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: self.monitor_gpu()
                ),
                # Reduce learning rate when stuck
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6
                ),
                # Save training history
                tf.keras.callbacks.CSVLogger(
                    "training_history.csv", separator=",", append=False
                ),
            ]

            # Calculate class weights
            total_samples = sum(self.class_counts.values())
            class_weights = {
                cls: total_samples / (len(self.class_counts) * count)
                for cls, count in self.class_counts.items()
            }

            logger.info("Starting training with GPU acceleration...")
            self.monitor_gpu()  # Initial GPU memory check

            # Train with weights and callbacks
            self.history = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=Config.EPOCHS,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1,
            )

            # Final GPU memory check
            self.monitor_gpu()

            # Save and plot
            self._plot_training_history()
            self.model.save("models/converted/final_model.keras")
            return True

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return False

    def _plot_training_history(self):
        """Plot training metrics"""
        import matplotlib.pyplot as plt

        # Plot accuracy
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history["accuracy"])
        plt.plot(self.history.history["val_accuracy"])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"])

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"])

        plt.tight_layout()
        plt.savefig("training_history.png")
        plt.close()

    def verify_data_pipeline(self):
        """Verify data pipeline integrity"""
        logger.info("Verifying data pipeline...")

        # Check a batch
        for images, labels in self.train_ds.take(1):
            logger.info(f"Batch shape: {images.shape}")
            logger.info(f"Labels shape: {labels.shape}")
            logger.info(
                f"Label range: {tf.reduce_min(labels)} to {tf.reduce_max(labels)}"
            )
            logger.info(
                f"Image value range: {tf.reduce_min(images)} to {tf.reduce_max(images)}"
            )

            # Save some sample images with their labels
            for i in range(min(5, len(images))):
                img = images[i].numpy()
                label = labels[i].numpy()
                tf.keras.utils.save_img(f"debug/sample_{i}_label_{label}.jpg", img)


def main():
    """Main training execution"""
    trainer = ModelTrainer()

    # Load datasets
    if not trainer.load_datasets():
        logger.error("Failed to load datasets. Exiting.")
        return

    # Setup model
    if not trainer.setup_model():
        logger.error("Failed to setup model. Exiting.")
        return

    # Train model
    if not trainer.train():
        logger.error("Training failed. Exiting.")
        return

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise
