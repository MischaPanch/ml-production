import logging
import sys
from typing import Union, Callable, List

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as f

args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

_log = logging.getLogger("sparkStuff")
df = spark.read.json("s3://<BUCKET>/raw/Gift_Cards_5.json.gz")


def manipulateNestedColumn(df, parentColName, childColName, mapping=None, newChildColName=None, addAsNewChild=False):
    #TODO: currently only works with exactly once nested columns

    def identity(x):
        return x

    if mapping is None:
        mapping = identity

    if newChildColName is None and addAsNewChild:
        raise ValueError(f"Cannot add the column {childColName} twice to {parentColName}")

    colToManipulate = df[f"{parentColName}.{childColName}"]
    if newChildColName is None:
        newChildColName = childColName
    if newChildColName.lower() == parentColName.lower():  # spark does not like that, it seems
        _log.warning(f"Modifying {newChildColName} to {newChildColName}_ due to name collision")
        newChildColName += "_"
    if addAsNewChild:
        childCols = [f"{parentColName}.*"]
    else:
        childCols = [f"{parentColName}.{child}" for child in df.select(f"{parentColName}.*").columns if child != childColName]

    return df.withColumn(newChildColName, mapping(colToManipulate)) \
        .withColumn(parentColName, f.struct(newChildColName, *childCols)) \
        .drop(newChildColName)


def manipulateNestedColumns(df, parentColName: str, childColNames: str, mappings: Union[Callable, List[Callable]] = None,
                            newChildColNames: List[str] = None, addAsNewChildren=False):
    if newChildColNames is None:
        newChildColNames = childColNames
    if not isinstance(mappings, list) and not isinstance(mappings, tuple):
        mappings = [mappings] * len(childColNames)
    assert len(newChildColNames) == len(childColNames) == len(mappings), \
        "Lengths of newChildColNames, childColNames and mappings have to match"

    for child, newChild, mapping in zip(childColNames, newChildColNames, mappings):
        df = manipulateNestedColumn(df, parentColName, child, mapping, newChildColName=newChild, addAsNewChild=addAsNewChildren)
    return df


def castToInt(col):
    return f.regexp_replace(col, " ", "").cast("int")


cleanDf = manipulateNestedColumn(df, "style", "Gift Amount:", castToInt, newChildColName="Gift_Amount")

cleanDf.printSchema()

cleanDf.write.mode("overwrite").parquet("s3://<BUCKET>/cleaned/glue/Gift_Cards")


job.commit()
