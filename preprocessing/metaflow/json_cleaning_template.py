from metaflow import FlowSpec, step, retry, Parameter, batch, conda, conda_base


@conda_base(python="3.7", libraries={'pandas': '1.0.1'})
class JsonCleaningFlow(FlowSpec):
    """
    A flow to clean jsons and persist them in S3

    The flow performs the following steps:
    1)

    """

    inputFile = Parameter("inputFile", help="file uri to read with pandas")
    outputDir = Parameter("outputDir",  help="uri of a directory", default="s3://...")

    @batch(cpu=1, memory=500)
    @step
    def start(self):
        """
        Load files from S3
        """
        self.prefixDataframeTuple = []
        self.next(self.clean_dataframe, foreach="prefixDataframeTuple")

    @batch(cpu=1, memory=500)
    @retry
    @step
    def clean_dataframe(self):
        """clean"""
        self.next(self.save_dataframe)

    @step
    def save_dataframe(self):
        """save"""
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
