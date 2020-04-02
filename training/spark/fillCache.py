import logging

import pandas as pd
from pyspark import Row, SparkContext
from pyspark.sql import functions as f, SparkSession
from sensai.data_transformation import DFTNormalisation
from sensai.featuregen import FeatureGeneratorFromColumnGenerator

from ..utils import ColumnGeneratorSentenceEncodings, BertBaseMeanEncodingProvider

_log = logging.getLogger(__name__)

if __name__ == "__main__":

    INPUT_FILE = "data/emr/cleaned/Gift_Cards_5.json.gz"

    logging.basicConfig(level=logging.INFO)
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc).builder.master("local").enableHiveSupport().getOrCreate()

    cleanGifts = spark.read.json(INPUT_FILE)

    flattenedDf = cleanGifts.select(f.concat_ws("_", "asin", "reviewerID", "unixReviewTime").alias("identifier"),
                                    "style.*", "overall", "reviewText")
    # flattenedDf.write.mode("overwrite").csv("flattenedGifts.csv", header=True)

    CACHE_PATH = "sentenceCache.sqlite"
    encodingProvider = BertBaseMeanEncodingProvider()

    def sentenceEmbeddingFeatureGeneratorFactory(persistCache=True):
        columnGen = ColumnGeneratorSentenceEncodings("reviewText", encodingProvider,
                                                     CACHE_PATH, persistCache=persistCache)
        return FeatureGeneratorFromColumnGenerator(columnGen,
                    normalisationRuleTemplate=DFTNormalisation.RuleTemplate(unsupported=True))


    def computeFeaturesFromRow(row: Row):
        if not isinstance(row.reviewText, str):
            return
        rowDict = row.asDict()
        rowPandasDf = pd.DataFrame(rowDict, index=[row.identifier])
        print(f"Computing entry for {row.identifier}")
        generator = sentenceEmbeddingFeatureGeneratorFactory()
        generator.generate(rowPandasDf)

    # filling the cache in parallel through multiple processes
    flattenedDf.foreach(computeFeaturesFromRow)

    # this will be passed to a training script
    flattenedPandasDf = flattenedDf.toPandas().set_index("identifier", drop=True).dropna()
