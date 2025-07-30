# utilities/s3_utils.py

import boto3
from botocore.exceptions import ClientError

# -------------------------
#     S3 Utilities
# -------------------------

def check_if_uri_exists(s3_uri: str) -> bool:
    """
    Check if an S3 object exists at the given URI.

    Params:
    - s3_uri (str): Full S3 URI (e.g., 's3://bucket/path/to/file')

    Returns:
    - bool: True if object exists, False otherwise
    """
    # Parse S3 URI
    if not s3_uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI format: {s3_uri}")

    uri_parts = s3_uri.replace('s3://', '').split('/', 1)
    bucket_name = uri_parts[0]
    object_key = uri_parts[1] if len(uri_parts) > 1 else ''

    s3_client = boto3.client('s3')

    try:
        s3_client.head_object(Bucket=bucket_name, Key=object_key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        else:
            # Re-raise other errors (permissions, etc.)
            raise
