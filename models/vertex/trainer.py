import tensorflow as tf
from google.cloud import aiplatform


class BirdClassifierTrainer:
    def __init__(self, config):
        self.config = config

    def create_model(self):
        """Create and compile model"""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.config.model.input_shape,
            alpha=self.config.model.alpha,
            include_top=False,
            weights="imagenet",
        )

        # Freeze base model
        base_model.trainable = False

        model = tf.keras.Sequential(
            [
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(
                    self.config.model.num_classes, activation="softmax"
                ),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def train_on_vertex(self, dataset):
        """Initialize and run Vertex AI training job"""
        job = aiplatform.CustomTrainingJob(
            display_name=self.config.model.name,
            script_path="trainer/task.py",
            container_uri="gcr.io/cloud-aiplatform/training/tf-gpu.2-12",
            requirements=["tensorflow==2.12.0"],
            machine_type=self.config.training.machine_type,
            accelerator_type=self.config.training.accelerator,
            accelerator_count=self.config.training.accelerator_count,
        )

        model = job.run(
            dataset=dataset,
            base_output_dir=f"gs://{self.config.project.bucket}/model_output",
        )

        return model
