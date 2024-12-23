import tensorflow as tf
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("chriamue/bird-species-dataset")

    # Print dataset info
    print("\nDataset Info:")
    print(dataset)

    # Define constants
    IMG_SIZE = 96
    BATCH_SIZE = 32

    def preprocess_image(example):
        # Convert image to RGB tensor and resize
        image = tf.image.resize(example["image"], (IMG_SIZE, IMG_SIZE))
        # Normalize pixel values to [0,1]
        image = tf.cast(image, tf.float32) / 255.0
        return image, example["label"]

    # Convert to TensorFlow datasets
    print("\nPreprocessing training data...")
    train_ds = tf.data.Dataset.from_tensor_slices(
        (dataset["train"]["image"], dataset["train"]["label"])
    )

    print("Preprocessing test data...")
    test_ds = tf.data.Dataset.from_tensor_slices(
        (dataset["test"]["image"], dataset["test"]["label"])
    )

    # Apply preprocessing and batching
    train_ds = train_ds.map(preprocess_image).batch(BATCH_SIZE)
    test_ds = test_ds.map(preprocess_image).batch(BATCH_SIZE)

    # Visualize a few examples
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            plt.title(f"Label: {labels[i].numpy()}")
            plt.axis("off")
    plt.savefig("sample_images.png")
    print("\nSample images saved as 'sample_images.png'")

    return train_ds, test_ds


if __name__ == "__main__":
    train_ds, test_ds = main()
