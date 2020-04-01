import os

import pandas as pd


def writeCleanedDf(inputPath, outputPath):
    df = pd.read_json(inputPath, lines=True)
    df["style"] = df["style"] \
        .apply(lambda x: {} if pd.isna(x) else x) \
        .apply(lambda x: {k.replace(" ", "_").replace(":", ""): int(v.strip()) for k, v in x.items()})
    outputPath = os.path.join(outputPath, os.path.basename(inputPath))
    df.to_json(outputPath, lines=True, orient="records")
