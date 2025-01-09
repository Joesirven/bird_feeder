from google.cloud import storage
from google.cloud import aiplatform
import tensorflow as tf


class DataPipeline:
    def __init__(self, config):
        self.config = config
        self.bucket_name = config.project.bucket

    def create_dataset(self):
        """Create Vertex AI dataset from GCS bucket"""
        dataset = aiplatform.ImageDataset.create(
            display_name=self.config.model.name,
            gcs_source=f"gs://{self.bucket_name}/processed/*",
            import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification,
        )

        return dataset

    def _augment(self, image, label):
        """Data augmentation function"""
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        return image, label

    def preprocess_for_training(self, dataset):
        """Create TF Dataset pipeline"""
        train_ds = dataset.as_dataset(split="train")

        # Augmentation and preprocessing
        train_ds = train_ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.shuffle(1000).batch(self.config.model.batch_size)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        return train_ds
