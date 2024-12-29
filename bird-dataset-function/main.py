import functions_framework
from google.cloud import storage
import kagglehub
import os
import json
from pathlib import Path


def setup_kaggle_credentials(kaggle_key, kaggle_secret):
    """Setup Kaggle credentials programmatically"""
    kaggle_dir = "/tmp/.kaggle"  # Use /tmp in Cloud Functions
    os.makedirs(kaggle_dir, exist_ok=True)

    # Create kaggle.json with provided credentials
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    credentials = {"username": kaggle_key, "key": kaggle_secret}

    with open(kaggle_json, "w") as f:
        json.dump(credentials, f)

    os.chmod(kaggle_json, 0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = kaggle_dir


def download_and_upload(bucket_name, kaggle_dataset):
    """Download dataset and upload to GCS"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Download dataset
    path = kagglehub.dataset_download(kaggle_dataset)

    # Upload all files to GCS
    local_path = Path(path)
    uploaded_files = []

    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            relative_path = str(file_path.relative_to(local_path))
            blob = bucket.blob(relative_path)
            blob.upload_from_filename(str(file_path))
            uploaded_files.append(relative_path)

    return uploaded_files


@functions_framework.http
def process_dataset(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
    """
    try:
        # Get credentials from environment variables
        kaggle_key = os.environ.get("KAGGLE_USERNAME")
        kaggle_secret = os.environ.get("KAGGLE_KEY")

        if not kaggle_key or not kaggle_secret:
            return "Error: Kaggle credentials not found in environment variables", 400

        # Setup credentials
        setup_kaggle_credentials(kaggle_key, kaggle_secret)

        # Constants
        BUCKET_NAME = "chriamue-bird-species-dataset"
        DATASET_NAME = "sharansmenon/inatbirds100k"

        # Process the dataset
        uploaded_files = download_and_upload(BUCKET_NAME, DATASET_NAME)

        return {
            "success": True,
            "message": f"Successfully processed {len(uploaded_files)} files",
            "files": uploaded_files,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}, 500
