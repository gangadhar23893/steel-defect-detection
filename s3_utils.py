import boto3

s3 = boto3.client('s3')

def download_file(bucket_name, object_name, local_path):
    s3.download_file(bucket_name, object_name, local_path)

    print(f"Downloaded {object_name} from {bucket_name}")


def list_files(bucket_name,prefix = ""):
    response = s3.list_objects_v2(
        Bucket=bucket_name,
        Prefix = prefix
    )
    files = []
    if "Contents" in response:
        for obj in response["Contents"]:
            files.append(obj["Key"])

    return files