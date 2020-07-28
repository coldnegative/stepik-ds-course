import io
import sys

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

MODEL_PATH = 'spark_ml_model'

def process(spark, train_data, test_data):
    df = spark.read.parquet(train_data)
    assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol='features')
    rf = RandomForestRegressor(labelCol='ctr')
    pipeline = Pipeline(stages=[assembler, rf])
    evaluator = RegressionEvaluator(labelCol='ctr')
    grid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, [3, 5, 7, 10])\
        .addGrid(rf.numTrees, [3, 6, 9, 12, 15, 18, 21, 24])\
        .addGrid(rf.subsamplingRate, [0.8, 0.9, 1])\
        .build()
    tvs = TrainValidationSplit(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=grid, trainRatio=0.8)
    model = tvs.fit(df)
    model.bestModel.save(MODEL_PATH)
    print(evaluator.evaluate(model.transform(spark.read.parquet(test_data))))

def main(argv):
    train_data = argv[0]
    print("Input path to train data: " + train_data)
    test_data = argv[1]
    print("Input path to test data: " + test_data)
    spark = _spark_session()
    process(spark, train_data, test_data)

def _spark_session():
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()

if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Train and test data are require.")
    else:
        main(arg)
