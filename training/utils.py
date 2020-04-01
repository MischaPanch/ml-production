import logging
from abc import ABC, abstractmethod
from typing import Union, List, Generic, TypeVar

import numpy as np
import sensai as sn
from textstat import textstat

T = TypeVar("T")


_log = logging.getLogger(__name__)


class SentenceEncoder(ABC):
    @abstractmethod
    def encode(self, sentence: str) -> np.ndarray:
        pass


class TransientInstanceProvider(ABC, Generic[T]):
    """Base class for providing transient instances of anything"""
    _log = _log.getChild(__qualname__)

    def __init__(self):
        self._instance = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_instance"] = None
        return d

    @abstractmethod
    def _getInstance(self) -> T:
        pass

    def getInstance(self) -> T:
        if self._instance is None:
            self._instance = self._getInstance()
        return self._instance


class BertBaseMeanSentenceEncoder(SentenceEncoder):
    def encode(self, sentence: str) -> np.ndarray:
        return self.embeddingModel.encode([sentence])[0]

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.embeddingModel = SentenceTransformer('bert-base-nli-mean-tokens')


class BertBaseMeanEncodingProvider(TransientInstanceProvider[BertBaseMeanSentenceEncoder]):
    def _getInstance(self):
        return BertBaseMeanSentenceEncoder()


class TextStatEncoder(SentenceEncoder):
    def encode(self, sentence: str) -> np.ndarray:
        if not isinstance(sentence, str):
            sentence = ''
        return np.array([
            textstat.flesch_reading_ease(sentence),
            textstat.syllable_count(sentence),
            textstat.text_standard(sentence, float_output=True),
            textstat.syllable_count(sentence),
            textstat.lexicon_count(sentence, removepunct=True)])


class TextStatEncodingProvider(TransientInstanceProvider[TextStatEncoder]):
    def _getInstance(self):
        return TextStatEncoder()


class ColumnGeneratorSentenceEncodings(sn.columngen.ColumnGeneratorCachedByIndex):
    """
    Encodes text column with label feature using the bert backbone.
    """
    def __init__(self, columnToEncode: Union[str, List[str]], encodingProvider: TransientInstanceProvider[SentenceEncoder],
                 sqliteCachePath: str, generatedColumnName: str = None, persistCache=False):
        """
        :param columnToEncode: name of column or list of columns that is to be encoded.
        :param encodingProvider: provider for sentence encoding
        :type generatedColumnName: if None, the generated column will be <columnToEncode>Encoded
        :param sqliteCachePath:
        """
        self.encodingProvider = encodingProvider
        self.columnToEncode = columnToEncode
        generatedColumnName = generatedColumnName if generatedColumnName is not None else f"{columnToEncode}Encoded"
        _cache = sn.util.cache.SqlitePersistentKeyValueCache(sqliteCachePath)

        super().__init__(generatedColumnName, _cache, persistCache=persistCache)

    def _generateValue(self, namedTuple):
        encoder = self.encodingProvider.getInstance()
        return encoder.encode(getattr(namedTuple, self.columnToEncode))

