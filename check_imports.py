import sys
import tensorflow as tf
import numpy as np

print("\nPython Executable:")
print(sys.executable)

print("\nTensorFlow Info:")
print(f"TensorFlow version: {tf.__version__}")
print(f"TensorFlow location: {tf.__file__}")

print("\nNumPy Info:")
print(f"NumPy version: {np.__version__}")
print(f"NumPy location: {np.__file__}")

print("\nPython Path:")
for path in sys.path:
    print(f"  {path}")
