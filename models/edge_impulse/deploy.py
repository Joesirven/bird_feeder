from edge_impulse_api import EdgeImpulseApi
import tensorflow as tf


class EdgeImpulseDeployer:
    def __init__(self, config):
        self.config = config
        self.ei_api = EdgeImpulseApi(api_key=config.edge_impulse.api_key)

    def optimize_model(self, model_path):
        """Optimize model for ESP32-S3"""
        converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        # Representative dataset for quantization
        def representative_dataset():
            for data in self.config.calibration_data.take(100):
                yield [data[0].numpy()]

        converter.representative_dataset = representative_dataset

        return converter.convert()

    def deploy(self, model_path):
        """Deploy to Edge Impulse"""
        # Create project
        project = self.ei_api.create_project(
            name=self.config.model.name, project_type="classification"
        )

        # Upload optimized model
        tflite_model = self.optimize_model(model_path)
        self.ei_api.upload_model(project_id=project.id, model_data=tflite_model)

        return project.id
