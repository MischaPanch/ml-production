import pickle

import pandas as pd
from sensai import InputOutputData
from sensai.data_transformation import DFTNormalisation
from sensai.evaluation import evalModelViaEvaluator
from sensai.featuregen import FeatureGeneratorFromColumnGenerator, flattenedFeatureGenerator, FeatureCollector
from sensai.torch import models

from training.utils import BertBaseMeanEncodingProvider, ColumnGeneratorSentenceEncodings


def sentenceEmbeddingFeatureGeneratorFactory(cachePath: str, persistCache=True):
    columnGen = ColumnGeneratorSentenceEncodings("reviewText", encodingProvider, cachePath, persistCache=persistCache)
    return FeatureGeneratorFromColumnGenerator(columnGen,
                                               normalisationRuleTemplate=DFTNormalisation.RuleTemplate(unsupported=True))


"""This script contains the training code snippet. Adjust it to your needs and insert it into you respective solution"""


if __name__ == "__main__":
    flattenedPandasDf: pd.DataFrame = ... # Load/insert the flattened dataframe from a previous step
    CACHE_PATH: str = ...

    # replace by a lightweight model for lambda
    reviewClassifier = models.MultiLayerPerceptronVectorClassificationModel(hiddenDims=[50, 50, 20], cuda=False, epochs=300)

    # add the feature generator that was previously used to fill the cache to the model
    # encodingProvider = TextStatEncodingProvider() # for lambda
    encodingProvider = BertBaseMeanEncodingProvider()
    reviewEncodingFeatureGen = sentenceEmbeddingFeatureGeneratorFactory(CACHE_PATH, persistCache=False)
    encodedReviewColName = reviewEncodingFeatureGen.columnGen.generatedColumnName
    flattenedSentenceEncodingsFeatureregen = flattenedFeatureGenerator(reviewEncodingFeatureGen,
                                                normalisationRuleTemplate=DFTNormalisation.RuleTemplate(skip=True))
    reviewFeatureCollector = FeatureCollector(flattenedSentenceEncodingsFeatureregen)
    reviewClassifier = reviewClassifier.withFeatureCollector(reviewFeatureCollector)

    # split off the targets and train
    targetDf = pd.DataFrame(flattenedPandasDf.pop("overall"))
    inputOutputData = InputOutputData(flattenedPandasDf, targetDf)
    evalModelViaEvaluator(reviewClassifier, inputOutputData, testFraction=0.01, plotTargetDistribution=True)

    # save model, load it and try predict as integration test
    with open("reviewClassifier-v1.pickle", 'wb') as f:
        pickle.dump(reviewClassifier, f)

    with open("reviewClassifier-v1.pickle", 'rb') as f:
        loadedModel = pickle.load(f)

    loadedModel.predict(flattenedPandasDf)
