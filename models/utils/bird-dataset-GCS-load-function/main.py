import functions_framework
from google.cloud import storage
import kagglehub
import os
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_kaggle_credentials(kaggle_key, kaggle_secret):
    """Setup Kaggle credentials programmatically"""
    try:
        kaggle_dir = "/tmp/.kaggle"  # Use /tmp in Cloud Functions
        os.makedirs(kaggle_dir, exist_ok=True)

        # Create kaggle.json with provided credentials
        kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
        credentials = {"username": kaggle_key, "key": kaggle_secret}

        with open(kaggle_json, "w") as f:
            json.dump(credentials, f)

        os.chmod(kaggle_json, 0o600)
        os.environ["KAGGLE_CONFIG_DIR"] = kaggle_dir
        logger.info("Kaggle credentials setup complete")

    except Exception as e:
        logger.error(f"Error setting up Kaggle credentials: {e}")
        raise


def download_and_upload(bucket_name, kaggle_dataset):
    """Download dataset and upload to GCS"""
    try:
        logger.info("Initializing storage client")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        logger.info(f"Starting download of dataset: {kaggle_dataset}")
        path = kagglehub.dataset_download(kaggle_dataset)
        logger.info(f"Dataset downloaded to: {path}")

        uploaded_files = []
        local_path = Path(path)

        logger.info("Starting upload to GCS")
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative_path = str(file_path.relative_to(local_path))
                blob = bucket.blob(relative_path)
                logger.info(f"Uploading: {relative_path}")
                blob.upload_from_filename(str(file_path))
                uploaded_files.append(relative_path)

        return uploaded_files

    except Exception as e:
        logger.error(f"Error in download_and_upload: {e}")
        raise


@functions_framework.http
def process_dataset(request):
    """HTTP Cloud Function."""
    try:
        logger.info("Starting process_dataset function")

        # Get credentials from environment variables
        kaggle_key = os.environ.get("KAGGLE_USERNAME")
        kaggle_secret = os.environ.get("KAGGLE_KEY")

        if not kaggle_key or not kaggle_secret:
            logger.error("Kaggle credentials not found in environment variables")
            return {
                "success": False,
                "error": "Kaggle credentials not found in environment variables",
            }, 400

        # Setup credentials
        setup_kaggle_credentials(kaggle_key, kaggle_secret)

        # Constants
        BUCKET_NAME = "bird-species-dataset"
        DATASET_NAME = "sharansmenon/inatbirds100k"

        # Process the dataset
        uploaded_files = download_and_upload(BUCKET_NAME, DATASET_NAME)

        logger.info(f"Successfully processed {len(uploaded_files)} files")
        return {
            "success": True,
            "message": f"Successfully processed {len(uploaded_files)} files",
            "files": uploaded_files,
        }

    except Exception as e:
        logger.error(f"Error in process_dataset: {e}")
        return {"success": False, "error": str(e)}, 500


def main():
    try:
        logger.info("Starting dataset processing")

        # Get credentials from environment variables
        kaggle_key = os.environ.get("KAGGLE_USERNAME")
        kaggle_secret = os.environ.get("KAGGLE_KEY")

        if not kaggle_key or not kaggle_secret:
            raise ValueError("Kaggle credentials not found in environment variables")

        # Setup credentials
        setup_kaggle_credentials(kaggle_key, kaggle_secret)

        # Constants
        BUCKET_NAME = "bird-species-dataset"
        DATASET_NAME = "sharansmenon/inatbirds100k"

        # Process the dataset
        uploaded_files = download_and_upload(BUCKET_NAME, DATASET_NAME)

        logger.info(f"Successfully processed {len(uploaded_files)} files")
        return 0

    except Exception as e:
        logger.error(f"Error in process_dataset: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
