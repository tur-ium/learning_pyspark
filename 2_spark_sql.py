# Importing Spark Session and SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkContext
import logging

logging.basicConfig(filename='log2.txt', filemode='w', level=logging.INFO)
# Get Spark Context (useful if need to use RDDs)
SparkContext.getOrCreate()

# Create a Spark Session (useful if need to use Spark data frames)
spark = SparkSession \
    .builder \
    .master("local") \
    .appName("Introduction to DataFrame") \
    .getOrCreate()

sc = SparkContext.getOrCreate()

# Define a schema for the data we expect
from pyspark.sql.types import StructType, StructField, StringType, DateType, IntegerType, BooleanType, NumericType, \
    DoubleType, CharType

titanic_schema = StructType([
    StructField('PassengerId',IntegerType(),False),
    StructField('Survived', BooleanType()),
    StructField('Pclass',IntegerType()),
    StructField('Name',StringType(),False),
    StructField('Sex',StringType()),
    StructField('Age',DoubleType()),
    StructField('SibSp',IntegerType()),
    StructField('Parch',IntegerType()),
    StructField('Ticket',StringType()),
    StructField('Fare',DoubleType()),
    StructField('Cabin',StringType()),
    StructField('Embarked',StringType())
])
df = spark.read.csv('data/titanic_dataset.csv', header=True,schema=titanic_schema)
# Show the top 5 rows
df.show(5)
df.printSchema()
print(df.dtypes)

# Create a new data frame using columns from the original
personal_info = df.select('Name', 'Age')
personal_info.show(5)
