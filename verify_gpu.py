# Create a test file: verify_gpu.py
import tensorflow as tf
import os

# Force Metal plugin to load
os.environ["DEVICE_NAME"] = "apple_metal"

print("Python version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# Test Metal Plugin
try:
    # Force GPU device context
    with tf.device("/GPU:0"):
        print("Testing GPU computation...")
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print("GPU test successful!")
        print(f"Computation shape: {c.shape}")

        # Verify we're actually using Metal
        print("\nDevice Placement:")
        print(tf.debugging.get_device_name(0))
except Exception as e:
    print(f"Error during GPU test: {e}")

print("\nSystem Information:")
print(f"TensorFlow version: {tf.__version__}")
print(f"Metal Plugin path: {tf.__file__}")
print("\nDevice Details:")
for device in tf.config.list_physical_devices():
    print(f"  {device.device_type}: {device.name}")
