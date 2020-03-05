import os

import pandas as pd
from metaflow import FlowSpec, step, retry, Parameter, batch, conda_base


@conda_base(python="3.7", libraries={'pandas': '1.0.1'})
class JsonCleaningFlow(FlowSpec):
    """
    A flow to clean jsons and persist them in S3

    The flow performs the following steps:
    1)

    """

    chunksize = Parameter("chunksize", default=int(1e3), help="Maximal number of rows to include into one process")
    inputFile = Parameter("inputFile", help="file uri to read with pandas",
                          default="s3://mlproduction-mpanchen/raw/Gift_Cards_5.json.gz")
    outputDir = Parameter("outputDir",  help="uri of a directory", default="s3://mlproduction-mpanchen/cleaned")

    @batch(cpu=1, memory=500)
    @step
    def start(self):
        """
        Load file from S3
        """
        # with metaflows internal s3 connector:
        #
        # with S3(s3root=self.bucket) as s3:
        #     print("trying to load from s3")
        #     s3obj = s3.get('raw/Gift_Cards_5.json.gz')
        #     print("Object found at", s3obj.url)
        #     self.file = gzip.open(BytesIO(s3obj.blob))

        self.dataframesWithPrefix = list(enumerate(pd.read_json(self.inputFile, lines=True, chunksize=self.chunksize)))
        self.next(self.clean_dataframe, foreach="dataframesWithPrefix")

    @batch(cpu=1, memory=500)
    @retry
    @step
    def clean_dataframe(self):
        """clean"""
        self.prefix, self.df = self.input
        self.df["style"] = self.df["style"] \
            .apply(lambda x: {} if pd.isna(x) else x) \
            .apply(lambda x: {k.replace(" ", "_").replace(":", ""): int(v.strip()) for k, v in x.items()})
        self.next(self.save_dataframe)

    @batch(cpu=1, memory=500)
    @retry
    @step
    def save_dataframe(self):
        """save"""
        outputFile = os.path.join(self.outputDir, f"{self.prefix}_{os.path.basename(self.inputFile)}")
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
