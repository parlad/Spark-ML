# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#Creating a spark session in order to have access to creating dataframes
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()


# %%
#Importing the algorithms and evaluator needed for creating the model and evaluating its performance
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# %%
#Loading in the data, printing the schema, and showing the top 20 rows
ecommerceData = spark.read.csv(r'Data/ServiceUsage.csv', header = True, inferSchema = True)
ecommerceData.printSchema()
ecommerceData.show()


# %%
#Immporting the vector libraries in order to transform the dataset
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# %%
#Feeding the dataframe into the vector assembler (transformer) and combining 4 columns into one column called "features"
assembler = VectorAssembler(inputCols = ['Avg Session Length', 'Time on App', 'Time on Website', 'Length of Membership'], outputCol = 'Features')
transformedEcommerceData = assembler.transform(ecommerceData)
transformedEcommerceData.show()


# %%
#Preparing the data for the model by only having two columns: the features and the column of known data we're trying to predict
finalData = transformedEcommerceData.select('Features', 'Yearly Amount Spent')
finalData.show()


# %%
#Splitting the data into training and testing sets by randomly choosing 70% of the rows for training and 30% of the rows for testing
trainingData, testingData = finalData.randomSplit([0.7, 0.3])



# %%
#Decision Tree Regression
decisionTree = DecisionTreeRegressor(featuresCol = "Features", labelCol = "Yearly Amount Spent", maxDepth = 15, maxBins = 32)
decisionTreeModel = decisionTree.fit(trainingData)
dtresults = decisionTreeModel.transform(testingData)
dtresults.select("Prediction", "Yearly Amount Spent", "Features")
dtresults.show()
#Using RMSE to evaluate the model
gbtevaluator = RegressionEvaluator(labelCol="Yearly Amount Spent", predictionCol="prediction", metricName="rmse")
gbtrmse = gbtevaluator.evaluate(dtresults)
print("Gradient-Boosted Tree RMSE: ", gbtrmse)
