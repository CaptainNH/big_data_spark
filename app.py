import time
import psutil
import os
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col as spark_col, when
from pyspark.sql.types import IntegerType, DoubleType, StringType
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def parse_arguments():
    parser = argparse.ArgumentParser(description='Student Depression Prediction')
    parser.add_argument('-o', '--optimized',
                        action='store_true',
                        help='Enable caching of training data for optimization')
    return parser.parse_args()


def main():
    args = parse_arguments()
    start_time = time.time()

    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("StudentDepressionAnalysis") \
        .master("spark://spark-master:7077") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    # Define schema explicitly
    from pyspark.sql.types import StructType, StructField
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("Gender", StringType(), True),
        StructField("Age", DoubleType(), True),
        StructField("City", StringType(), True),
        StructField("Profession", StringType(), True),
        StructField("Academic Pressure", DoubleType(), True),
        StructField("Work Pressure", DoubleType(), True),
        StructField("CGPA", DoubleType(), True),
        StructField("Study Satisfaction", DoubleType(), True),
        StructField("Job Satisfaction", DoubleType(), True),
        StructField("Sleep Duration", StringType(), True),
        StructField("Dietary Habits", StringType(), True),
        StructField("Degree", StringType(), True),
        StructField("Have you ever had suicidal thoughts ?", StringType(), True),
        StructField("Work/Study Hours", DoubleType(), True),
        StructField("Financial Stress", DoubleType(), True),
        StructField("Family History of Mental Illness", StringType(), True),
        StructField("Depression", IntegerType(), True)
    ])

    # Load the dataset with explicit schema
    df = spark.read.csv("hdfs://namenode:9000/data/student_depression_dataset.csv",
                        header=True,
                        schema=schema)

    # Data preprocessing
    def preprocess_data(df):
        # Convert Depression to binary (1 for depressed, 0 for not depressed)
        df = df.withColumn("Depression", spark_col("Depression").cast("integer"))

        # Handle missing values
        numeric_cols = ["Age", "Academic Pressure", "Work Pressure", "CGPA",
                        "Study Satisfaction", "Job Satisfaction", "Work/Study Hours",
                        "Financial Stress"]
        for col in numeric_cols:
            df = df.na.fill(0, subset=[col])

        # Convert categorical columns to numerical
        categorical_cols = [
            "Gender",
            "City",
            "Profession",
            "Dietary Habits",
            "Degree",
            "Have you ever had suicidal thoughts ?",
            "Family History of Mental Illness"
        ]

        # Convert Sleep Duration (string) to numerical
        df = df.withColumn("Sleep Duration",
                           when(df["Sleep Duration"] == "Less than 5 hours", 4)
                           .when(df["Sleep Duration"] == "5-6 hours", 5)
                           .when(df["Sleep Duration"] == "6-7 hours", 6)
                           .when(df["Sleep Duration"] == "7-8 hours", 7)
                           .when(df["Sleep Duration"] == "More than 8 hours", 8)
                           .otherwise(0))

        indexers = [StringIndexer(inputCol=column, outputCol=column + "_index")
                    for column in categorical_cols]
        pipeline = Pipeline(stages=indexers)
        return pipeline.fit(df).transform(df)

    processed_df = preprocess_data(df)

    # Split data into train and test
    train_df, test_df = processed_df.randomSplit([0.8, 0.2], seed=42)

    # Cache the training data if optimized
    if args.optimized:
        train_df.cache()
        print("Optimization enabled: Training data cached in memory")

    # Select features and target
    target_col = "Depression"
    numeric_cols = [
        "Age",
        "Academic Pressure",
        "Work Pressure",
        "CGPA",
        "Study Satisfaction",
        "Job Satisfaction",
        "Work/Study Hours",
        "Financial Stress",
        "Sleep Duration"
    ]
    categorical_index_cols = [col + "_index" for col in [
        "Gender",
        "City",
        "Profession",
        "Dietary Habits",
        "Degree",
        "Have you ever had suicidal thoughts ?",
        "Family History of Mental Illness"
    ]]
    all_features = numeric_cols + categorical_index_cols

    # Create pipeline
    pipeline = Pipeline(stages=[
        VectorAssembler(inputCols=all_features, outputCol="features_unscaled"),
        StandardScaler(inputCol="features_unscaled", outputCol="features", withStd=True, withMean=True),
        GBTClassifier(featuresCol="features", labelCol=target_col, maxIter=100, seed=42)
    ])

    # Train model
    model = pipeline.fit(train_df)

    # Evaluate model
    predictions = model.transform(test_df)
    accuracy = MulticlassClassificationEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="accuracy"
    ).evaluate(predictions)

    f1 = MulticlassClassificationEvaluator(
        labelCol=target_col, predictionCol="prediction", metricName="f1"
    ).evaluate(predictions)

    # Print results
    print("\n=== Model Evaluation ===")
    print("Accuracy: {:.2%}".format(accuracy))
    print("F1 Score: {:.2%}".format(f1))
    print("\n=== Performance Metrics ===")
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))
    print("Memory usage: {:.2f} MB".format(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)))

    spark.stop()


if __name__ == "__main__":
    main()