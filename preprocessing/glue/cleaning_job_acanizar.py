import sys
from functools import partial

from awsglue.dynamicframe import DynamicFrame
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

def flatten(dyn):
    """Flatten dynamic frame. Nested a|b columns become a single column named 'a.b'"""
    return UnnestFrame.apply(dyn)

def unflatten(dyn):
    """Nest columns with '.' in its names. Single column 'a.b' becomes nested a|b."""
    fieldMap = dyn.schema().field_map
    mapping = []
    for colName, field in fieldMap.items():
        fieldType = field.dataType.jsonValue()["dataType"]
        # In mapping names containing '.' must be escaped with ``. Not escaped '.' are interpreted as nestings
        mapping.append((f"`{colName}`", fieldType, colName, fieldType))

    return dyn.apply_mapping(mapping)

def cleanColumns(dynFlat):
    """Remove whitespaces and ':' from column names"""
    fieldMap = dynFlat.schema().field_map
    for colName in fieldMap:
        cleanName = colName.replace(":", "").replace(" ", "")
        dynFlat = dynFlat.rename_field(f"`{colName}`", f"`{cleanName}`")
    return dynFlat

def stripAndCastToInt(dynFlat, colPrefix):
    """Strip values and cast to int for column names matching the given prefix"""
    fieldMap = dynFlat.schema().field_map
    matchedCols = [col for col in fieldMap if col.startswith(colPrefix)]

    def stripRec(rec, cols):
        for col in cols:
            val = rec[col]
            rec[col] = int(val.strip())
        return rec

    stripDyn = dynFlat.map(partial(stripRec, cols=matchedCols))
    return stripDyn

#dyn = glueContext.create_dynamic_frame.from_catalog(database = "<GLUE_DB>", table_name = "gift_cards_5_json_gz", transformation_ctx = "dyn")
dyn = DynamicFrame.fromDF(spark.read.json("s3://<BUCKET>/raw/Gift_Cards_5.json.gz"), glueContext, "dyn")

dynFlat = flatten(dyn)
dynFlat = cleanColumns(dynFlat)
dynFlat = stripAndCastToInt(dynFlat, "style")
dynClean = unflatten(dynFlat)

datasink = glueContext.write_dynamic_frame.from_options(frame=dynClean, connection_type="s3", connection_options={"path": "s3://<BUCKET>/clean"}, format="json", transformation_ctx="datasink")
job.commit()