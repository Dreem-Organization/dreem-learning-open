import boto3
from botocore import UNSIGNED
from botocore.client import Config
import tqdm
from dreem_learning_open.settings import DODH_SETTINGS, DODO_SETTINGS

client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

bucket_objects = client.list_objects(Bucket='dreem-dod-o')["Contents"]
print("\n Downloading H5 files and annotations from S3 for DOD-O")
for bucket_object in tqdm.tqdm(bucket_objects):
    filename = bucket_object["Key"]
    client.download_file(
        Bucket="dreem-dod-o",
        Key=filename,
        Filename=DODO_SETTINGS["h5_directory"] + "/{}".format(filename)
    )

print("\n Downloading H5 files and annotations from S3 for DOD-H")
for bucket_object in tqdm.tqdm(bucket_objects):
    filename = bucket_object["Key"]
    client.download_file(
        Bucket="dreem-dod-h",
        Key=filename,
        Filename=DODH_SETTINGS["h5_directory"] + "/{}".format(filename)
    )
