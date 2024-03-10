# Importing SparkSession and SParkContext
import logging
import time

from pyspark.sql import SparkSession
from pyspark import SparkContext

s=time.time()

# Creating a SparkContext
sc = SparkContext.getOrCreate()

# Creating a Spark session
spark = SparkSession \
    .builder \
    .appName("ML Tuning") \
    .getOrCreate()

# Loading the raw database
df_full = spark.read.csv('data/YearPredictionMSD.txt', header=False)

# We infer the right types from the columns
from pyspark.sql.functions import col
exprs = [col(c).cast("double") for c in df_full.columns[1:13]]

df_casted = df_full.select(df_full._c0.cast('int'),
                           *exprs)

# Finally, for the sake of speed of calculations,
# we will only process an extract from the database in this exercise
df = df_casted.sample(False, .1, seed = 222)

df.sample(False, .001, seed = 222).toPandas()

from pyspark.ml.linalg import DenseVector

sample_size_for_ten_rows = 10./df.count()
df.sample(sample_size_for_ten_rows,seed=222).toPandas()

rdd_ml = df.rdd.map(lambda x: (x[0],DenseVector(x[1:])))
df_ml = spark.createDataFrame(rdd_ml, ['label','features'])
df_ml.sample(sample_size_for_ten_rows,seed=222).toPandas()

train, test = df_ml.randomSplit([0.8,0.2],seed=12)

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(labelCol='label',featuresCol='features')
linear_model = lr.fit(train)
print(linear_model)

"""
Parameter setting by cross validation

    Once the parameter grid is constructed, it is possible to vary the parameters in this grid in order to obtain the best possible parameters using several methods. The most commonly used method is cross-validation.

    k-fold cross validation is a very effective method to significantly improve the robustness of a model. Its principle is as follows:

        Division of the original sample into k samples
        Selection of one of the k samples (hold-out) as a validation set and the other (k-1) samples will constitute the learning set
        Estimation of the error by calculating a test, measure or model performance score on the test sample
        Repeat the operation by selecting another validation sample from among the (k-1) samples that have not yet been used for model validation

    The operation is repeated k times so that in the end each sub-sample was used exactly once as a validation set. The average of the k performance scores is finally calculated to estimate the prediction error.
    """

# Create a parameter grid, called param_grid, containing the values 0, 0.5 and 1 for the parameters:
#     regParam
#     elasticNetParam
from pyspark.ml.tuning import  ParamGridBuilder
param_grid = (ParamGridBuilder().
              addGrid(lr.regParam,[0, 0.5, 1]).
              addGrid(lr.elasticNetParam, [0, 0.5, 1]) # Curious about the elasticNetParam? -> https://machinelearningcompass.com/machine_learning_models/elastic_net_regression/
              .build())

# An elastic net parameter is a combination of two techniques - rigde and lasso

# Create an evaluator taking into account the evaluation metric r2
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(predictionCol='prediction',labelCol='label',metricName='r2')

from pyspark.ml.tuning import CrossValidator

cv=CrossValidator(estimator=lr,estimatorParamMaps=param_grid,evaluator=evaluator,numFolds=3)

cv_model = cv.fit(train)
e=time.time()
print(f'Done in {e-s:.2f}s')

# Apply model on train and test sets
pred_train = cv_model.transform(train)
pred_test = cv_model.transform(test)

# Evaluate root mean square error metic on predicted test set results using our evaluator from earlier
rmse_test = evaluator.setMetricName('rmse').evaluate(pred_test)
print(f'RMSE on test set = {rmse_test:.2f}')

from pprint import pprint

print(f'The coefficients for the best model are: {cv_model.bestModel.coefficients}')
# Spark will even go and explain the parameters for the best model!!
pprint(cv_model.bestModel.explainParams())