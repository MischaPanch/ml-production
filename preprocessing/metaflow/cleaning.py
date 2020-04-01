import os

import pandas as pd
from metaflow import FlowSpec, step, Parameter


# noinspection PyTypeChecker
# @conda_base(python="3.7", libraries={'pandas': '1.0.1'})
class JsonCleaningFlow(FlowSpec):
    """
    A flow to clean jsons and persist them in S3

    The flow performs the following steps:
    1)

    """

    chunksize = Parameter("chunksize", default=int(1e3), help="Maximal number of rows to include into one process")
    inputFile = Parameter("inputFile", help="file uri to read with pandas (local or s3)",
                          default="data/raw/Gift_Cards_5.json.gz")
    outputDir = Parameter("outputDir",  help="uri of a directory (local or s3)", default="data/metaflow/cleaned")

    @step
    def start(self):
        """
        Load file (from S3 or local)
        """
        self.dataframes = list(pd.read_json(self.inputFile, lines=True, chunksize=self.chunksize))
        self.next(self.clean_dataframe, foreach="dataframes")

    @step
    def clean_dataframe(self):
        """clean the style columns"""
        self.df: pd.DataFrame = self.input
        self.df["style"] = self.df["style"] \
            .apply(lambda x: {} if pd.isna(x) else x) \
            .apply(lambda x: {k.replace(" ", "_").replace(":", ""): int(v.strip()) for k, v in x.items()})
        self.next(self.save_dataframe)

    @step
    def save_dataframe(self):
        """save as json"""
        outputFile = os.path.join(self.outputDir, f"{self.index}_{os.path.basename(self.inputFile)}")
        if not outputFile.startswith("s3://"):
            os.makedirs(os.path.dirname(outputFile), exist_ok=True)
        self.df.to_json(outputFile, lines=True, orient="records")
        self.next(self.join)

    @step
    def join(self, inputs):
        """ This does nothing but is unfortunately required by metaflow"""
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass


if __name__ == '__main__':
    JsonCleaningFlow()
