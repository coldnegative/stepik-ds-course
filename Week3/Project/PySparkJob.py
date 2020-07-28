import io
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff
from pyspark.sql import functions as F

def writer(df, target_path):
    train, test, validate = df.randomSplit([0.5, 0.25, 0.25])
    train.write.parquet('%s/'%target_path+'train/')
    test.write.parquet('%s/'%target_path+'test/')
    validate.write.parquet('%s/'%target_path+'validate/')

def process(spark, input_file, target_path):
    df = spark.read.parquet(input_file)
    cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
    df = df.withColumn('is_cpm', F.when(col('ad_cost_type')=='CPM', 1).otherwise(0)).withColumn('is_cpc', F.when(col('ad_cost_type')=='CPC', 1).otherwise(0))\
        .drop('ad_cost_type','client_union_id','compaign_union_id','platform')\
        .groupBy('ad_id','target_audience_count','has_video','is_cpm','is_cpc','ad_cost').agg(
            (datediff(F.max('date'),F.min('date'))+1).alias('day_count'),
            (cnt_cond(F.col('event') == 'click')/cnt_cond(F.col('event') == 'view')).alias('CTR')
        )
    writer(df,target_path)

def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    target_path = argv[1]
    print("Target path: " + target_path)
    spark = _spark_session()
    process(spark, input_path, target_path)


def _spark_session():
    return SparkSession.builder.appName('PySparkJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)