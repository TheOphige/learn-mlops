# Databricks notebook source
from pyspark.sql import SparkSession

# COMMAND ----------

spark=SparkSession.builder.appName('structured_streaming').getOrCreate()

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

from pyspark.sql.types import *

# COMMAND ----------

df_1=spark.createDataFrame([("XN203",'FB',300,30), ("XN201",'Twitter',10,19),("XN202",'Insta',500,45)], ["user_id","app","time_in_secs","age"]).write.csv ("csv_folder",mode='append')

# COMMAND ----------

schema=StructType().add("user_id","string").add("app","string").add("time_in_secs", "integer").add("age", "integer")

# COMMAND ----------

data=spark.readStream.option("sep", ",").schema(schema).table("csv_folder")

# COMMAND ----------

data.printSchema()

# COMMAND ----------

app_count=data.groupBy('app').count()

# COMMAND ----------

query=(app_count.writeStream.queryName('count_query').outputMode('complete').format('memory').start())

# COMMAND ----------

spark.sql("select * from count_query ").toPandas().head(5)

# COMMAND ----------

