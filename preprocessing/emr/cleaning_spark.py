import logging
import os
import re
from typing import List, Callable, Union

from pyspark import Row, SparkContext
from pyspark.sql import functions as f, SparkSession

_log = logging.getLogger(__name__)


def manipulateNestedColumn(df, parentColName, childColName, mapping=None, newChildColName=None, addAsNewChild=False):
    # TODO: currently only works with exactly once nested columns

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


def manipulateNestedThroughRdd(df, parentColName, childColName, mapping,
        newChildColName=None, addAsNewChild=False, newChildType=None):
    # TODO: currently only works with exactly once nested columns
    if newChildColName is None and addAsNewChild:
        raise ValueError(f"Cannot add the column {childColName} twice to {parentColName}")

    _sc = SparkContext.getOrCreate()
    _spark = SparkSession(_sc).builder.master("local").getOrCreate()

    def adjustRow(row: Row) -> Row:
        rowDict = row.asDict()
        parentRow = rowDict.pop(parentColName)
        if parentRow is None:
            return row

        parentDict = parentRow.asDict()
        newChildValue = mapping(parentDict[childColName])
        if not addAsNewChild:
            parentDict.pop(childColName)
        parentDict[newChildColName] = newChildValue
        rowDict[parentColName] = Row(**parentDict)
        return Row(**rowDict)

    # adjust the schema of the resulting df - unfortunately spark cannot infer it automatically
    childSchemaString = f"{parentColName}:struct<{childColName}"
    newChildSchemaString = f"{parentColName}:struct<{newChildColName}"
    schemaString = df.schema.simpleString()
    if newChildType is not None:
        newSchemaString = re.sub(f"{childSchemaString}:.*?>", f"{newChildSchemaString}:{newChildType}>", schemaString)
    else:
        newSchemaString = re.sub(childSchemaString, newChildSchemaString, schemaString)
    return _spark.createDataFrame(df.rdd.map(adjustRow), newSchemaString)


if __name__ == "__main__":
    # This should be passed through argparse and spark submit in a real scenario
    INPUT_FILE = "data/raw/Gift_Cards_5.json.gz"
    OUTPUT_FILE = "data/emr/cleaned/Gift_Cards_5.json.gz"

    logging.basicConfig(level=logging.INFO)
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc).builder.master("local").enableHiveSupport().getOrCreate()

    def castToInt(col):
        return f.trim(col).cast("int")

    # read, clean and write
    gifts = spark.read.json(INPUT_FILE)
    cleanGifts = manipulateNestedColumn(gifts, "style", "Gift Amount:", castToInt, newChildColName="Gift_Amount")
    if not OUTPUT_FILE.startswith("s3://"):
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    cleanGifts.write.mode("overwrite").json(OUTPUT_FILE)
