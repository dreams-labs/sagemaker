"""
ETLs for interactions that relate to S3, including uploads of local datasources.
"""
import logging
from pathlib import Path
import boto3

# set up logger at the module level
logger = logging.getLogger(__name__)



# -----------------------------
#         ETL Functions
# -----------------------------

def upload_folder_to_s3(
    local_path: str,
    bucket_name: str,
    s3_target_folder: str = ""
) -> None:
    """
    Upload all files from local folder to S3 bucket with confirmation prompt.
    Uses local folder name as final S3 directory within s3_target_folder.

    Params:
    - local_path (str): Path to local folder
    - bucket_name (str): S3 bucket name
    - s3_target_folder (str): S3 parent folder path

    Raises:
    - FileNotFoundError: If local_path doesn't exist
    - ValueError: If local_path contains no files
    - ClientError: If S3 bucket doesn't exist
    """
    local_folder = Path(local_path)
    expected_extensions = {'.parquet', '.csv', '.json'}

    # Check if path exists
    if not local_folder.exists():
        raise FileNotFoundError(f"Local path does not exist: {local_path}")

    # Get all files and calculate total size
    all_files = [f for f in local_folder.rglob("*") if f.is_file()]

    if not all_files:
        raise ValueError(f"No files found in {local_path}")

    # Build final S3 path using local folder name
    local_folder_name = local_folder.name
    s3_full_path = f"{s3_target_folder}/{local_folder_name}" if s3_target_folder else local_folder_name

    total_size_bytes = sum(f.stat().st_size for f in all_files)
    total_size_gb = total_size_bytes / (1024**3)

    # Check for unexpected file types
    unexpected_files = [f for f in all_files if f.suffix.lower() not in expected_extensions]
    if unexpected_files:
        logger.warning(f"Found {len(unexpected_files)} files with unexpected extensions: "
                        f"{[f.suffix for f in unexpected_files]}")

    # Confirmation prompt
    logger.info(f"Ready to upload {len(all_files)} files with total size {total_size_gb:.2f}GB")
    logger.info(f"Target: s3://{bucket_name}/{s3_full_path}")
    confirmation = input("Proceed with upload? (y/N): ")

    if confirmation.lower() != 'y':
        logger.info("Upload cancelled")
        return

    # Initialize S3 client and verify bucket exists
    s3_client = boto3.client('s3')
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except Exception as e:
        raise ValueError(f"S3 bucket '{bucket_name}' does not exist or is not accessible") from e

    for file_path in all_files:
        relative_path = file_path.relative_to(local_folder)
        s3_key = f"{s3_full_path}/{relative_path}".replace("\\", "/")

        logger.info(f"Uploading {file_path.name} -> s3://{bucket_name}/{s3_key}")
        s3_client.upload_file(str(file_path), bucket_name, s3_key)

    logger.info(f"Upload complete: {len(all_files)} files")
