# step3_ml.py
# Trains three ML models on the Silver layer Wikipedia edit data
# Run after step2 has collected data in /tmp/wiki_output/silver/
#
# Run with:
#   PYSPARK_PYTHON=/path/to/.venv/bin/python3 \
#   spark-submit step3_ml.py

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    RegressionEvaluator,
    ClusteringEvaluator,
)
from pyspark.sql.window import Window

# Paths - must match what step2 wrote to
SILVER_PATH = "/data/silver/*.parquet"
MODEL_PATH  = "/models"
RANDOM_SEED     = 42
TEST_SPLIT      = 0.2


# Creates and returns a Spark session in batch mode for ML training
def createSparkSession():
    spark = (
        SparkSession.builder
        .appName("WikipediaML")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# Loads Silver parquet data and prints row count
def loadSilverData(spark):
    print("Loading silver data from:", SILVER_PATH)
    df = spark.read.parquet(SILVER_PATH)
    totalRows = df.count()
    print("Total rows loaded:", totalRows)
    if totalRows < 500:
        print("Warning: low row count, models may not be accurate")
    return df


# Filters to article edits only and prepares features for ML
# Uses camelCase column names from our step2_spark_streaming.py
def prepareFeatures(df):
    return (
        df
        .filter(F.col("namespace") == 0)
        .filter(F.col("type") == "edit")
        .select(
            "wiki", "user", "bot", "isAnon",
            "byte_delta", "title_len", "commentLen",
            "minor", "hourOfDay", "dayOfWeek",
            "editSizeCategory", "wikiLang", "latencyMs",
        )
        .fillna({
            "byte_delta":  0,
            "title_len":   0,
            "commentLen":  0,
            "latencyMs":   0,
            "hourOfDay":   0,
            "dayOfWeek":   1,
        })
        # Cast booleans to int for ML input
        .withColumn("isBot",   F.col("bot").cast("int"))
        .withColumn("isAnon",  F.col("isAnon").cast("int"))
        .withColumn("isMinor", F.col("minor").cast("int"))
        .withColumn("absDelta", F.abs(F.col("byte_delta")))
        .dropna(subset=["bot"])
    )


# Prints basic RDD statistics about the feature data
def printRddStats(featureDf):
    print("\n--- RDD Descriptive Statistics ---")
    statsRdd = featureDf.rdd.map(lambda r: (
        r["byte_delta"] or 0,
        r["commentLen"] or 0,
        r["isBot"] or 0,
    ))
    totalCount  = statsRdd.count()
    totalBytes  = statsRdd.map(lambda x: x[0]).sum()
    botCount    = statsRdd.filter(lambda x: x[2] == 1).count()
    maxDelta    = statsRdd.map(lambda x: x[0]).max()
    minDelta    = statsRdd.map(lambda x: x[0]).min()

    print("  Total edits        :", totalCount)
    print("  Total bytes changed:", totalBytes)
    print("  Bot edits          :", botCount, f"({100*botCount/max(totalCount,1):.1f}%)")
    print("  Max byte delta     :", maxDelta)
    print("  Min byte delta     :", minDelta)


# Trains a RandomForest classifier to detect bot vs human edits
# Returns the trained model and AUC score
def trainBotClassifier(featureDf):
    print("\n--- Model A: Bot vs Human Classifier ---")

    featureCols = [
        "byte_delta", "absDelta", "title_len", "commentLen",
        "isAnon", "isMinor", "hourOfDay", "dayOfWeek",
    ]

    assembler = VectorAssembler(
        inputCols=featureCols,
        outputCol="featuresRaw",
        handleInvalid="skip",
    )
    scaler = StandardScaler(
        inputCol="featuresRaw",
        outputCol="features",
        withMean=True,
        withStd=True,
    )
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="isBot",
        numTrees=50,
        maxDepth=6,
        seed=RANDOM_SEED,
    )

    pipeline = Pipeline(stages=[assembler, scaler, rf])
    trainDf, testDf = featureDf.randomSplit([1 - TEST_SPLIT, TEST_SPLIT], seed=RANDOM_SEED)

    print("Training rows:", trainDf.count(), "| Test rows:", testDf.count())
    model = pipeline.fit(trainDf)

    predictions = model.transform(testDf)
    evaluator   = BinaryClassificationEvaluator(labelCol="isBot", metricName="areaUnderROC")
    auc         = evaluator.evaluate(predictions)
    print("Bot Classifier AUC-ROC:", round(auc, 4))

    # Print feature importances
    rfModel = model.stages[-1]
    print("\nFeature importances:")
    for name, imp in sorted(zip(featureCols, rfModel.featureImportances), key=lambda x: x[1], reverse=True):
        print(f"  {name:<20s} {imp:.4f}")

    # Show sample predictions
    print("\nSample predictions:")
    predictions.select("user", "isBot", "prediction", "byte_delta", "commentLen").show(10, truncate=False)

    # Save model to disk
    savePath = f"{MODEL_PATH}/bot_classifier"
    model.write().overwrite().save(savePath)
    print("Model saved to:", savePath)
    return model, auc


# Trains a KMeans clustering model to detect vandalism anomalies
# Returns the trained model and silhouette score
def trainVandalismDetector(featureDf):
    print("\n--- Model B: Vandalism Anomaly Detector (KMeans) ---")

    featureCols = [
        "byte_delta", "absDelta", "title_len",
        "commentLen", "isAnon", "isMinor",
    ]

    assembler = VectorAssembler(
        inputCols=featureCols,
        outputCol="featuresRaw",
        handleInvalid="skip",
    )
    scaler = MinMaxScaler(inputCol="featuresRaw", outputCol="features")
    kmeans = KMeans(
        k=5,
        featuresCol="features",
        predictionCol="cluster",
        seed=RANDOM_SEED,
        maxIter=20,
    )

    pipeline    = Pipeline(stages=[assembler, scaler, kmeans])
    model       = pipeline.fit(featureDf)
    predictions = model.transform(featureDf)

    evaluator  = ClusteringEvaluator(predictionCol="cluster", featuresCol="features")
    silhouette = evaluator.evaluate(predictions)
    print("KMeans silhouette score:", round(silhouette, 4))

    # Print cluster profiles
    print("\nCluster profiles:")
    predictions.groupBy("cluster").agg(
        F.count("*")        .alias("count"),
        F.avg("byte_delta") .alias("avgDelta"),
        F.avg("commentLen") .alias("avgComment"),
        F.avg("isAnon")     .alias("anonRatio"),
        F.avg("isMinor")    .alias("minorRatio"),
    ).orderBy("cluster").show()

    # Save model to disk
    savePath = f"{MODEL_PATH}/vandalism_kmeans"
    model.write().overwrite().save(savePath)
    print("Model saved to:", savePath)
    return model, silhouette


# Trains a GBT regressor to forecast edit volume using lag features
# Returns the trained model and RMSE score
def trainVolumeForecaster(silverDf):
    print("\n--- Model C: Edit Volume Forecaster (GBTRegressor) ---")

    # Aggregate edits into 1-minute windows per wiki
    windowedDf = (
        silverDf
        .filter(F.col("namespace") == 0)
        .withColumn("minuteWindow", F.window("eventTime", "1 minute"))
        .groupBy("wiki", "minuteWindow")
        .agg(
            F.count("*")                        .alias("editCount"),
            F.avg("byte_delta")                 .alias("avgDelta"),
            F.sum(F.col("bot").cast("int"))     .alias("botEdits"),
            F.sum(F.col("isAnon").cast("int"))  .alias("anonEdits"),
        )
        .withColumn("windowStart", F.col("minuteWindow.start"))
        .withColumn("hourOfDay",   F.hour("windowStart"))
        .withColumn("dayOfWeek",   F.dayofweek("windowStart"))
        .drop("minuteWindow")
        .fillna(0)
    )

    # Add lag features using Spark window functions
    w = Window.partitionBy("wiki").orderBy("windowStart")
    laggedDf = (
        windowedDf
        .withColumn("editLag1",  F.lag("editCount", 1).over(w))
        .withColumn("editLag2",  F.lag("editCount", 2).over(w))
        .withColumn("editLag3",  F.lag("editCount", 3).over(w))
        .withColumn("deltaLag1", F.lag("avgDelta",  1).over(w))
        .withColumn("botLag1",   F.lag("botEdits",  1).over(w))
        .dropna()
    )

    rowCount = laggedDf.count()
    if rowCount < 100:
        print("Not enough windowed data for forecasting, skipping Model C")
        return None, None

    featureCols = ["editLag1", "editLag2", "editLag3", "deltaLag1", "botLag1", "hourOfDay", "dayOfWeek"]
    assembler   = VectorAssembler(inputCols=featureCols, outputCol="features", handleInvalid="skip")
    gbt         = GBTRegressor(featuresCol="features", labelCol="editCount", maxIter=30, maxDepth=4, seed=RANDOM_SEED)

    pipeline            = Pipeline(stages=[assembler, gbt])
    trainDf, testDf     = laggedDf.randomSplit([1 - TEST_SPLIT, TEST_SPLIT], seed=RANDOM_SEED)
    model               = pipeline.fit(trainDf)
    predictions         = model.transform(testDf)

    rmse = RegressionEvaluator(labelCol="editCount", predictionCol="prediction", metricName="rmse").evaluate(predictions)
    r2   = RegressionEvaluator(labelCol="editCount", predictionCol="prediction", metricName="r2").evaluate(predictions)
    print("Volume Forecaster RMSE:", round(rmse, 4), "| R2:", round(r2, 4))

    # Print feature importances
    gbtModel = model.stages[-1]
    print("\nFeature importances:")
    for name, imp in sorted(zip(featureCols, gbtModel.featureImportances), key=lambda x: x[1], reverse=True):
        print(f"  {name:<15s} {imp:.4f}")

    # Show predicted vs actual
    print("\nPredicted vs actual edit volume:")
    predictions.select("wiki", "windowStart", "editCount", "prediction", "hourOfDay").orderBy("windowStart").show(15)

    # Save model to disk
    savePath = f"{MODEL_PATH}/volume_forecaster"
    model.write().overwrite().save(savePath)
    print("Model saved to:", savePath)
    return model, rmse


# Main function - loads data, runs all three models, prints summary
def main():
    os.makedirs(MODEL_PATH, exist_ok=True)

    spark     = createSparkSession()
    silverDf  = loadSilverData(spark)
    featureDf = prepareFeatures(silverDf)
    featureDf.cache()

    printRddStats(featureDf)

    modelA, auc        = trainBotClassifier(featureDf)
    modelB, silhouette = trainVandalismDetector(featureDf)
    modelC, rmse       = trainVolumeForecaster(silverDf)

    print("\n=== ML TRAINING SUMMARY ===")
    print("Model A - Bot Classifier   AUC       :", round(auc, 4))
    print("Model B - Vandalism KMeans Silhouette :", round(silhouette, 4))
    if rmse:
        print("Model C - Volume GBT       RMSE      :", round(rmse, 4))
    print("Models saved to:", MODEL_PATH)

    spark.stop()


if __name__ == "__main__":
    main()