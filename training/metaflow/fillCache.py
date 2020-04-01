import os
from glob import glob

import pandas as pd
from metaflow import FlowSpec, step, Parameter
from sensai.data_transformation import DFTNormalisation
from sensai.featuregen import FeatureGeneratorFromColumnGenerator

from training.utils import ColumnGeneratorSentenceEncodings, BertBaseMeanEncodingProvider

encodingProvider = BertBaseMeanEncodingProvider()


def sentenceEmbeddingFeatureGeneratorFactory(cachePath: str, persistCache=True):
    columnGen = ColumnGeneratorSentenceEncodings("reviewText", encodingProvider, cachePath, persistCache=persistCache)
    return FeatureGeneratorFromColumnGenerator(columnGen,
                normalisationRuleTemplate=DFTNormalisation.RuleTemplate(unsupported=True))


# noinspection PyTypeChecker
# @conda_base(python="3.7", libraries={'pandas': '1.0.1'})
class CacheFillingFlow(FlowSpec):
    """
    A flow to clean jsons and persist them in S3

    The flow performs the following steps:
    1)

    """

    chunksize = Parameter("chunksize", default=int(1e3), help="Maximal number of rows to include into one process")
    inputDir = Parameter("inputDir", help="path to directory containing the cleaned json files",
                         default="data/metaflow/cleaned")
    cachePath = Parameter("cachePath",  help="path to sqlite cache. In production should be replaced by a proper SQL Database",
                          default="sentenceCache.sqlite")

    @step
    def start(self):
        """
        Load file (from S3 or local)
        """
        if self.inputDir.startswith("s3://"):
            raise NotImplementedError(f"Reading files from s3 is left as an exercise for the user")
        else:
            self.inputFiles = list(glob(os.path.join(self.inputDir, "*.json.gz")))

        self.next(self.generate_features, foreach="inputFiles")

    @step
    def generate_features(self):
        """Generate features with the purpose of filling the cache in parallel"""
        # TODO: this should connect to a real database instead of sqlite
        featureGen = sentenceEmbeddingFeatureGeneratorFactory(f"{self.index}_{self.cachePath}")
        df: pd.DataFrame = pd.read_json(self.input, lines=True)

        df = df[df["style"] != {}]  # drop rows with empty style dict
        df["identifier"] = df.apply(lambda row: "_".join(map(str, [row.asin, row.reviewerID, row.unixReviewTime])), axis=1)
        df.set_index("identifier", drop=True, inplace=True)
        df["giftAmount"] = df["style"].apply(lambda x: x["Gift_Amount"])
        df.drop(columns="style", inplace=True)
        featureGen.generate(df)
        self.next(self.join)

    @step
    def join(self, inputs):
        """ This does nothing but is unfortunately required by metaflow"""
        # TODO: either merge the sqlite databases or use a real one in the previous step
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass


if __name__ == '__main__':
    CacheFillingFlow()
