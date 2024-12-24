import tensorflow as tf
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from models.training.process import main as process_data
from models.training.config import Config

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
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)],
        )
        logger.info(f"Metal GPU enabled: {physical_devices}")
    else:
        logger.warning("No Metal GPU found")
except Exception as e:
    logger.warning(f"Error configuring Metal GPU: {e}")


class ModelTrainer:
    def __init__(self):
        self.model = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.history = None
        self.checkpoint_dir = Path(Config.CHECKPOINT_PATH)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Optimize dataset performance
        self.strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Number of devices: {self.strategy.num_replicas_in_sync}")

    def load_datasets(self):
        """Load cached preprocessed datasets with fixed dimensions"""
        logger.info("Loading cached preprocessed datasets...")
        try:
            cache_dir = Path("data/cache")

            def load_and_prepare(split_name):
                ds = tf.data.Dataset.load(str(cache_dir / f"{split_name}_processed"))
                return (
                    ds.map(
                        # Ensure correct dimensions
                        lambda x, y: (
                            tf.ensure_shape(
                                tf.cast(x, tf.float32) / 255.0,
                                [96, 96, 3],  # Force correct shape
                            ),
                            y,
                        ),
                        num_parallel_calls=tf.data.AUTOTUNE,
                    )
                    .batch(Config.BATCH_SIZE, drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE)
                )

            # Load datasets
            self.train_ds = load_and_prepare("train")
            self.val_ds = load_and_prepare("validation")
            self.test_ds = load_and_prepare("test")

            # Verify shapes
            for x, y in self.train_ds.take(1):
                logger.info(f"Input shape: {x.shape}, Label shape: {y.shape}")

            return True

        except Exception as e:
            logger.error(f"Failed to load cached datasets: {str(e)}")
            return False

    def setup_model(self):
        """Initialize EfficientNet-B2 with M1 optimizations"""
        logger.info("Setting up EfficientNet-B2 model...")

        try:
            # Set mixed precision policy before model creation
            tf.keras.mixed_precision.set_global_policy(
                "float32"
            )  # Changed from mixed_float16

            # Create model without distribution strategy for simplicity
            # Remove the strategy scope since we're running on single device

            # 1. Create base model
            base_model = tf.keras.applications.EfficientNetB2(
                include_top=False,
                weights="imagenet",
                input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3),
            )

            # 2. Freeze base model
            base_model.trainable = False

            # 3. Create full model architecture
            inputs = tf.keras.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3))
            x = base_model(inputs)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(512, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            outputs = tf.keras.layers.Dense(Config.NUM_CLASSES, activation="softmax")(x)

            # 4. Create the full model
            self.model = tf.keras.Model(inputs, outputs)

            # 5. Compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            # 6. Print model summary
            self.model.summary(print_fn=logger.info)
            logger.info("Model setup complete")
            return True

        except Exception as e:
            logger.error(f"Failed to setup model: {str(e)}")
            return False

    def train(self):
        """Execute training loop with M1 optimizations"""
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.checkpoint_dir / "best_model.keras"),
                save_best_only=True,
                monitor="val_accuracy",
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                profile_batch="100,120",  # Enable profiling
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=2
            ),
        ]

        try:
            self.history = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=Config.EPOCHS,
                callbacks=callbacks,
            )

            self.model.save("models/converted/final_model.keras")
            return True
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return False


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
