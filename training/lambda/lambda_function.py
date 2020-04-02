import urllib.parse

import pandas as pd
from sensai.data_transformation import DFTNormalisation
from sensai.featuregen import FeatureGeneratorFromColumnGenerator
from utils import ColumnGeneratorSentenceEncodings, TextStatEncodingProvider

encodingProvider = TextStatEncodingProvider()


def sentenceEmbeddingFeatureGeneratorFactory(cachePath: str, persistCache=True):
    columnGen = ColumnGeneratorSentenceEncodings("reviewText", encodingProvider, cachePath, persistCache=persistCache)
    return FeatureGeneratorFromColumnGenerator(columnGen,
                                               normalisationRuleTemplate=DFTNormalisation.RuleTemplate(unsupported=True))


def lambda_handler(event, context):
    # print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    inputPath = f"s3://{bucket}/{key}"
    CACHE_PATH = "sentenceCache.sqlite" # TODO: connect to a real database that allows concurrency etc (e.g. MySQL)
    print(f"bucket is {bucket}, key is {key}")
    try:
        featureGen = sentenceEmbeddingFeatureGeneratorFactory(CACHE_PATH)
        df = pd.read_json(inputPath, lines=True)

        df = df[df["style"] != {}]  # drop rows with empty style dict
        df["identifier"] = df.apply(lambda row: "_".join(map(str, [row.asin, row.reviewerID, row.unixReviewTime])), axis=1)
        df.set_index("identifier", drop=True, inplace=True)
        df["giftAmount"] = df["style"].apply(lambda x: x["Gift_Amount"])
        df.drop(columns="style", inplace=True)
        featureGen.generate(df)
    except Exception as e:
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e
