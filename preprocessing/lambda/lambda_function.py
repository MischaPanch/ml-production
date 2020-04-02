import urllib.parse
from cleaning import writeCleanedDf
import urllib.parse

from cleaning import writeCleanedDf


def lambda_handler(event, context):
    # print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    inputPath = f"s3://{bucket}/{key}"
    outputPath = f"s3://{bucket}/lambda/cleaned/{os.path.basename(key)}"
    print(f"bucket is {bucket}, key is {key}")
    try:
        writeCleanedDf(inputPath, outputPath)
    except Exception as e:
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e
