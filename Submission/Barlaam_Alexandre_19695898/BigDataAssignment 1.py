
from pyspark.sql.functions import min,max,mean,percentile_approx,variance,count
import matplotlib
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, LinearSVC, MultilayerPerceptronClassifier
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

nPS = spark.read.options(header='True', inferSchema='True').csv("nuclear_plants_small_dataset.csv")
nPS.show(10)
nPS.printSchema()


#Task 1 - Missing Datasets

#calculates the null values in each column
for i in range(13):
    value = nPS.where(nPS[nPS.schema[i].name].isNull()).count() 
    print('column {0:0d} has {1:0} null values' .format(i + 1, value))


#Task 2 - Summary Statistics & Box Plots

#calculates the summary statistics
for i in range(1, 13):
    column = nPS.schema[i].name
    print(column)
    Vals = nPS.groupBy("Status").agg(min(column),max(column),mean(column).alias("mean({0})".format(column)), percentile_approx(column, 0.5).alias("median({0})".format(column)), variance(column).alias("variance_values({0})".format(column)),).show(truncate=False)    

#converts the pyspark df to a pandas df
nPS_pd = nPS.toPandas()
nPS_pd.head()


#plots two boxplots for each status for each feature
nPS_pd.boxplot(by='Status', column=['Power_range_sensor_1'])
nPS_pd.boxplot(by='Status', column=['Power_range_sensor_2'])
nPS_pd.boxplot(by='Status', column=['Power_range_sensor_3 '])
nPS_pd.boxplot(by='Status', column=['Power_range_sensor_4'])
nPS_pd.boxplot(by='Status', column=['Pressure _sensor_1'])
nPS_pd.boxplot(by='Status', column=['Pressure _sensor_2'])
nPS_pd.boxplot(by='Status', column=['Pressure _sensor_3'])
nPS_pd.boxplot(by='Status', column=['Pressure _sensor_4'])
nPS_pd.boxplot(by='Status', column=['Vibration_sensor_1'])
nPS_pd.boxplot(by='Status', column=['Vibration_sensor_2'])
nPS_pd.boxplot(by='Status', column=['Vibration_sensor_3'])
nPS_pd.boxplot(by='Status', column=['Vibration_sensor_4'])


#Task 3 - Correlation Matrix

#plots the correlation matrix and shows it
CorrMat = nPS_pd.corr()
CorrMat


#Task 4 - Data Split(70% training set & 30% test set)

#splits the data into a 70% training set and 30% test set
(train, test) = nPS.randomSplit([0.7,0.3]) 

#counts the total values and the values for each group in each set and in the original data 
print(nPS.count())
print(nPS.where(nPS.Status == "Normal").count())
print(nPS.where(nPS.Status == "Abnormal").count())
print()
print(train.count())
print(train.where(nPS.Status == "Normal").count())
print(train.where(nPS.Status == "Abnormal").count())
print()
print(test.count())
print(test.where(nPS.Status == "Normal").count())
print(test.where(nPS.Status == "Abnormal").count())


#Task 5 - Decision Tree, Support Vector Machine Model & Artificial Neural Network

#defines the string indexer
indexer = StringIndexer(inputCol = "Status", outputCol = "statusIndex")

features = ['Power_range_sensor_1','Power_range_sensor_2','Power_range_sensor_3 ','Power_range_sensor_4','Pressure _sensor_1','Pressure _sensor_2','Pressure _sensor_3','Pressure _sensor_4','Vibration_sensor_1','Vibration_sensor_2','Vibration_sensor_3','Vibration_sensor_4']

#defines the vector assembler
VA = VectorAssembler(inputCols = features, outputCol = 'features')

#defines the vector indexer
VI = VectorIndexer(inputCol = "features", outputCol = "featuresIndex",maxCategories=4)

#defines the evaluation methods
Accuracy = MulticlassClassificationEvaluator(labelCol="statusIndex", predictionCol="prediction", metricName="accuracy")
Sensitivity = MulticlassClassificationEvaluator(labelCol="statusIndex", predictionCol="prediction", metricName="truePositiveRateByLabel")
Specificity = MulticlassClassificationEvaluator(labelCol="statusIndex", predictionCol="prediction", metricName="falsePositiveRateByLabel")

#Decision Tree
#defines the classification method
dTC = DecisionTreeClassifier(labelCol="statusIndex", featuresCol="featuresIndex")

#defines the ML workflow
pipelineDT = Pipeline(stages=[indexer, VA, VI, dTC])

#trains the classifier
modelDT = pipelineDT.fit(train) 

#tests the classifier
predictionsDT = modelDT.transform(test)

predictionsDT.printSchema()
predictionsDT.select("statusIndex", "features","prediction").show(10)

#evaluates the classifier 
accDT = Accuracy.evaluate(predictionsDT)
sens_DT = Sensitivity.evaluate(predictionsDT)
spec_DT = Specificity.evaluate(predictionsDT)
print("Error Rate: {0:0}".format(1.0 - accDT))
print("Sensitivity: {0:0}".format(sens_DT))
print("Specificity: {0:0}".format(spec_DT))


#Support Vector Machine
lSVC = LinearSVC(labelCol="statusIndex", featuresCol="featuresIndex", maxIter=10, regParam=0.1)

pipelineVM = Pipeline(stages=[indexer, VA, VI, lSVC])

modelVM = pipelineVM.fit(train)

predictionsVM = modelVM.transform(test)

predictionsVM.printSchema()
predictionsVM.select("statusIndex", "features", "prediction").show(10)

accVM = Accuracy.evaluate(predictionsVM)
sens_VM = Sensitivity.evaluate(predictionsVM)
spec_VM = Specificity.evaluate(predictionsVM)
print("Error Rate: {0:0}".format(1.0 - accVM))
print("Sensitivity: {0:0}".format(sens_VM))
print("Specificity: {0:0}".format(spec_VM))


#Artificial Neural Network
mPC = MultilayerPerceptronClassifier(labelCol="statusIndex", featuresCol="featuresIndex", layers=[len(features), 20, 10, 2], seed=123)

pipelineANN = Pipeline(stages=[indexer, VA, VI, mPC])

modelANN = pipelineANN.fit(train)

predictionsANN = modelANN.transform(test)

predictionsANN.printSchema()
predictionsANN.select("statusIndex", "features", "prediction").show(10)

accANN = Accuracy.evaluate(predictionsANN)
sens_ANN = Sensitivity.evaluate(predictionsANN)
spec_ANN = Specificity.evaluate(predictionsANN)
print("Error Rate: {0:0}".format(1.0 - accANN))
print("Sensitivity: {0:0}".format(sens_ANN))
print("Specificity: {0:0}".format(spec_ANN))


nPB = spark.read.options(header='True', inferSchema='True').csv("nuclear_plants_big_dataset.csv")
nPB.show(10)
nPB.printSchema()


#Task 8 - MapReduce 

#converts the df to an rdd
numbers = nPB.rdd
#maps the rdd
numbersMap = numbers.map(lambda x:(x,1)) 
