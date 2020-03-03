from itertools import zip_longest
import gzip
import os
from typing import Iterable, Callable, Any
import logging

_log = logging.getLogger(__name__)


class FileSplitter:
    def __init__(self):
        self.outputExt = ".gz"
        self.inputFileOpener = gzip.GzipFile
        self.outputFileOpener = lambda x: gzip.GzipFile(x, mode="wb")

    @staticmethod
    def _grouper(iterable: Iterable, maxChunkLines: int):
        iteratorReferences = [iter(iterable)] * maxChunkLines
        return zip_longest(*iteratorReferences, fillvalue=None)

    def getOutputFilePath(self, inputFile: str, postfix: str = None, outputDir: str = None):
        """
        Create an output path based on input. If the FileSplitter's outputExt field is None, the output's extension will
        be the same as the input's

        :param inputFile:
        :param postfix:
        :param outputDir: if None, will be basedir of input file

        :return: path of the form outputDir/<inputFileName><postfix>.<extension>
        """
        inputDir = os.path.abspath(os.path.dirname(inputFile))
        inputFileName, inputExt = os.path.splitext(os.path.basename(inputFile))

        outputDir = outputDir if outputDir is not None else inputDir
        outputDir = os.path.join(outputDir, inputFileName)
        outputFileName = f"{inputFileName}{postfix}"
        outputExt = self.outputExt if self.outputExt is not None else inputExt
        return os.path.join(outputDir, outputFileName+outputExt)

    def withInputFileOpener(self, fileOpener: Callable[[str], Iterable]):
        self.inputFileOpener = fileOpener
        return self

    def withOutputFileOpener(self, fileOpener: Callable[[str], Any], outputExt: str = None):
        """

        :param fileOpener: callable returning an open file
        :param outputExt: None, the output's extension will be the same as the input's
        """
        self.outputFileOpener = fileOpener
        self.outputExt = outputExt
        return self

    def split(self, inputFile: str, maxChunkLines: int, outputDir: str = None):
        with self.inputFileOpener(inputFile) as fIn:
            for i, chunkLines in enumerate(self._grouper(fIn, maxChunkLines)):
                outputPath = self.getOutputFilePath(inputFile, postfix=f"_{i}", outputDir=outputDir)
                os.makedirs(os.path.dirname(outputPath), exist_ok=True)
                with self.outputFileOpener(outputPath) as fOut:
                    _log.debug(f"Saving {outputPath}")
                    try:
                        fOut.writelines(chunkLines)
                    except TypeError:  # This can happen when we have reached the last line
                        pass
            _log.info(f"Successfully split {inputFile} into {i+1} chunks located at {os.path.dirname(outputPath)}")
