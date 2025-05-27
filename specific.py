import boto3
from botocore.config import Config

my_config = Config(
    region_name='us-east-1',
    retries={'max_attempts': 10}
)

client = boto3.client('pricing', config=my_config)


response = client.describe_services(ServiceCode='AmazonS3')
print(response)