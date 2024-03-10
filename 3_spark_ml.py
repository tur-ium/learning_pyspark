"""
Learning Aim: To understand the Spark Machine Learning library

Machine learning problem:
Is there a correlation between the characteristics of songs, and the year of release?
Second, can we predict it?

We use the Year Prediction MSD (Million Song Dataset) for this task, which has audio characteristics of
515345 (mostly) "Western" commercial songs released from 1922 to 2011.
"""
# Importing Spark Session and SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkContext
import logging

logging.basicConfig(filename='log3.txt', filemode='w', level=logging.INFO)
# Get Spark Context (useful if need to use RDDs)
SparkContext.getOrCreate()

# Building a Spark Session
spark = SparkSession.builder.appName("Starting with Spark ML").getOrCreate()

df_raw = spark.read.csv('data/YearPredictionMSD.txt')
df_raw

# Convert datatypes
from pyspark.sql.functions import col
exprs = [col(x).cast("double") for x in df_raw.columns[1:]]
exprs.insert(0,col(df_raw.columns[0]).cast("int"))
df = df_raw.select(*exprs)
df.printSchema()
sample_size_for_ten_rows = 10./df.count()
df.sample(sample_size_for_ten_rows,seed=12).toPandas()

# Do we have nulls?
from pyspark.sql.functions import isnan, when, count, col

null_values = df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).show()
# zeros in all columns indicate  that there are no nulls, that would interfere machine learning

# Convert database to svmlib format

# We need to convert the database into a dataframe with 2 columns
# - label (the variable to be predicted)
# - featrues (the explanatory variables)
# We'll convert the explanatory variables into one column by turning it into a vector (DenseVector)
from pyspark.ml.linalg import DenseVector

rdd_ml = df.rdd.map(lambda x: (x[0],DenseVector(x[1:])))
df_ml = spark.createDataFrame(rdd_ml, ['label','features'])
df_ml.sample(sample_size_for_ten_rows,seed=12).toPandas()

"""
From the documentation on UC Irvine's data set website
You should respect the following train / test split:
train: first 463,715 examples
test: last 51,630 examples
It avoids the 'producer effect' by making sure no song
from a given artist ends up in both the train and test set.
"""
train_row_count = 463715.
train_perc=train_row_count/df.count()
test_row_count = 51630.
test_perc=test_row_count/df.count()
train, test = df_ml.randomSplit([0.7,0.3],seed=12)

# What is the best regression parameter?
reg_param = 0.0
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(labelCol='label',featuresCol='features',regParam=reg_param)
linearModel = lr.fit(train)
print(linearModel)

predicted_years = linearModel.transform(test)
predicted_years.sample(0.001).toPandas()

# Summaries of fitting to train data
rmse = linearModel.summary.rootMeanSquaredError
r2 = linearModel.summary.r2

print(f'RMSE of LinearRegression model = {rmse} (smaller is better)')
print(f'r2 value of LinearRegression model = {r2} (smaller is better)')

# How accurate is it, considering the test data?
spark.stop()