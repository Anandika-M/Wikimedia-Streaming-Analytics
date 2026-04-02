# # step2_spark_streaming.py
# # Reads Wikipedia edit events from Kafka and saves them to Bronze, Silver, Gold layers
# #
# # Run with:
# #   PYSPARK_PYTHON=/path/to/.venv/bin/python3 \
# #   spark-submit \
# #     --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0 \
# #     step2_spark_streaming.py

# import os
# from pyspark.sql import SparkSession
# from pyspark.sql import functions as F
# from pyspark.sql.types import StringType

# from schema import wikiEventSchema

# # Output paths on your local Mac filesystem
# BRONZE_PATH     = "/tmp/wiki_output/bronze"
# SILVER_PATH     = "/tmp/wiki_output/silver"
# GOLD_PATH       = "/tmp/wiki_output/gold"
# CHECKPOINT_BASE = "/tmp/wiki_checkpoints"

# KAFKA_SERVER    = "localhost:9092"
# KAFKA_TOPIC     = "wikipedia"
# TRIGGER_TIME    = "15 seconds"


# # Creates output directories if they do not exist yet
# def createOutputDirs():
#     for path in [BRONZE_PATH, SILVER_PATH, GOLD_PATH]:
#         os.makedirs(path, exist_ok=True)


# # Creates and returns the Spark session with Kafka support
# def createSparkSession():
#     spark = (
#         SparkSession.builder
#         .appName("WikipediaPipeline")
#         .master("local[*]")
#         .config("spark.sql.shuffle.partitions", "4")
#         .config("spark.streaming.stopGracefullyOnShutdown", "true")
#         .getOrCreate()
#     )
#     spark.sparkContext.setLogLevel("WARN")
#     return spark


# # Reads the raw binary stream from Kafka and returns a DataFrame
# def readKafkaStream(spark):
#     return (
#         spark.readStream
#         .format("kafka")
#         .option("kafka.bootstrap.servers", KAFKA_SERVER)
#         .option("subscribe", KAFKA_TOPIC)
#         .option("startingOffsets", "latest")
#         .option("failOnDataLoss", "false")
#         .option("maxOffsetsPerTrigger", 500)
#         .load()
#     )


# # Parses raw Kafka bytes into structured columns using wikiEventSchema
# # This is the Bronze layer - raw data with proper types, no business logic
# def parseToBronze(rawStream):
#     return (
#         rawStream
#         .select(
#             F.from_json(
#                 F.col("value").cast(StringType()),
#                 wikiEventSchema
#             ).alias("e")
#         )
#         .select("e.*")
#         .filter(F.col("wiki").isNotNull())
#     )


# # Cleans and enriches the Bronze data into the Silver layer
# # Adds derived columns useful for analysis and ML
# def bronzeToSilver(bronzeDf):
#     return (
#         bronzeDf
#         # Convert unix timestamp to readable datetime
#         .withColumn("eventTime",
#             F.to_timestamp(F.from_unixtime(F.col("timestamp"))))

#         # Convert ingestion_ts (milliseconds) to datetime
#         .withColumn("ingestionTime",
#             F.to_timestamp((F.col("ingestion_ts") / 1000).cast("long")))

#         # Calculate lag between event and when producer received it
#         .withColumn("latencyMs",
#             F.col("ingestion_ts") - (F.col("timestamp") * 1000))

#         # Flag anonymous users by checking if username looks like an IP address
#         .withColumn("isAnon",
#             F.col("user").rlike(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"))

#         # Categorize the edit size based on byte change
#         .withColumn("editSizeCategory",
#             F.when(F.col("byte_delta").isNull(),  "unknown")
#             .when(F.col("byte_delta") >  5000,    "large_add")
#             .when(F.col("byte_delta") >   500,    "small_add")
#             .when(F.col("byte_delta") >=    0,    "tiny")
#             .when(F.col("byte_delta") >  -500,    "small_del")
#             .otherwise(                           "large_del"))

#         # Length of the edit comment (useful NLP feature)
#         .withColumn("commentLen",
#             F.length(F.coalesce(F.col("comment"), F.lit(""))))

#         # Hour and day extracted from event time (temporal features)
#         .withColumn("hourOfDay",  F.hour(F.col("eventTime")))
#         .withColumn("dayOfWeek",  F.dayofweek(F.col("eventTime")))

#         # Extract language code from wiki name e.g. "enwiki" -> "en"
#         .withColumn("wikiLang",
#             F.regexp_extract(F.col("wiki"), r"^([a-z]+)", 1))

#         # Drop rows with invalid timestamps or unknown event types
#         .filter(F.col("timestamp") > 0)
#         .filter(F.col("type").isin("edit", "new", "categorize", "log"))
#     )


# # Aggregates Silver data into 5-minute windows per wiki
# # This is the Gold layer - ready for dashboards and reporting
# def silverToGold(silverDf):
#     return (
#         silverDf
#         .filter(F.col("type") == "edit")
#         .filter(F.col("namespace") == 0)
#         .withWatermark("eventTime", "1 minute")
#         .groupBy(
#             F.window("eventTime", "5 minutes"),
#             F.col("wiki")
#         )
#         .agg(
#             F.count("*")                            .alias("totalEdits"),
#             F.sum(F.col("bot").cast("int"))         .alias("botEdits"),
#             F.sum(F.col("isAnon").cast("int"))      .alias("anonEdits"),
#             F.avg("byte_delta")                     .alias("avgByteDelta"),
#             F.approx_count_distinct("user")         .alias("uniqueEditors"),
#             F.avg("commentLen")                     .alias("avgCommentLen"),
#         )
#         # Calculate bot and anon ratios for downstream ML
#         .withColumn("botRatio",
#             F.col("botEdits") / F.col("totalEdits"))
#         .withColumn("anonRatio",
#             F.col("anonEdits") / F.col("totalEdits"))
#         # Simple vandalism signal: high anon ratio with many edits
#         .withColumn("vandalismFlag",
#             (F.col("anonRatio") > 0.3) & (F.col("totalEdits") > 5))
#         .withColumn("windowStart", F.col("window.start"))
#         .withColumn("windowEnd",   F.col("window.end"))
#         .drop("window")
#     )


# # Writes a streaming DataFrame to Parquet files on disk
# def writeStream(streamDf, outputPath, checkpointName, outputMode="append"):
#     return (
#         streamDf.writeStream
#         .outputMode(outputMode)
#         .format("parquet")
#         .option("path", outputPath)
#         .option("checkpointLocation", f"{CHECKPOINT_BASE}/{checkpointName}")
#         .trigger(processingTime=TRIGGER_TIME)
#         .start()
#     )


# # Prints a summary of each micro-batch to the console for monitoring
# def printBatchSummary(batchDf, batchId):
#     if batchDf.isEmpty():
#         return
#     totalRows = batchDf.count()
#     editCount = batchDf.filter(F.col("type") == "edit").count()
#     botCount  = batchDf.filter(F.col("bot") == True).count()
#     print(f"\nBatch {batchId} | total={totalRows} | edits={editCount} | bots={botCount}")


# # Main function - wires up the full pipeline and starts all streaming queries
# def main():
#     createOutputDirs()
#     spark = createSparkSession()

#     print("Pipeline starting - reading from Kafka topic:", KAFKA_TOPIC)

#     # Build the three layers
#     rawStream  = readKafkaStream(spark)
#     bronzeDf   = parseToBronze(rawStream)
#     silverDf   = bronzeToSilver(bronzeDf)
#     goldDf     = silverToGold(silverDf)

#     # Start writing Bronze layer to disk
#     bronzeQuery = writeStream(bronzeDf, BRONZE_PATH, "bronze")
#     print("Bronze layer writing to:", BRONZE_PATH)

#     # Start writing Silver layer to disk
#     silverQuery = writeStream(silverDf, SILVER_PATH, "silver")
#     print("Silver layer writing to:", SILVER_PATH)

#     # Start writing Gold layer to disk (uses append + watermark)
#     goldQuery = writeStream(goldDf, GOLD_PATH, "gold")
#     print("Gold layer writing to:", GOLD_PATH)

#     # Print a live summary to console every batch
#     monitorQuery = (
#         silverDf.writeStream
#         .foreachBatch(printBatchSummary)
#         .option("checkpointLocation", f"{CHECKPOINT_BASE}/monitor")
#         .trigger(processingTime=TRIGGER_TIME)
#         .start()
#     )

#     print("All queries running. Press Ctrl+C to stop.")

#     # Keep all queries running until manually stopped
#     try:
#         spark.streams.awaitAnyTermination()
#     except KeyboardInterrupt:
#         print("Stopping pipeline...")
#         for query in spark.streams.active:
#             query.stop()
#         spark.stop()
#         print("Pipeline stopped.")


# if __name__ == "__main__":
#     main()

# step2_spark_streaming.py
# Reads Wikipedia edit events from Kafka, processes Bronze/Silver/Gold layers
# AND writes Silver data live to Cassandra in each micro-batch
#
# Run with:
#   PYSPARK_PYTHON=/path/to/.venv/bin/python3 \
#   spark-submit \
#     --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1 \
#     step2_spark_streaming.py

import os
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from cassandra.cluster import Cluster
from cassandra.policies import RoundRobinPolicy

from schema import wikiEventSchema

# Output paths
BRONZE_PATH     = "/tmp/wiki_output/bronze"
SILVER_PATH     = "/tmp/wiki_output/silver"
GOLD_PATH       = "/tmp/wiki_output/gold"
CHECKPOINT_BASE = "/tmp/wiki_checkpoints"

# Kafka settings
KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC  = "wikipedia"
TRIGGER_TIME = "15 seconds"

# Cassandra settings
CASSANDRA_HOST = "127.0.0.1"
CASSANDRA_PORT = 9042
CASSANDRA_KS   = "wikipedia"


# Creates output directories if they do not exist
def createOutputDirs():
    for path in [BRONZE_PATH, SILVER_PATH, GOLD_PATH]:
        os.makedirs(path, exist_ok=True)


# Creates and returns the Spark session
def createSparkSession():
    spark = (
        SparkSession.builder
        .appName("WikipediaPipeline")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# Reads the raw binary stream from Kafka
def readKafkaStream(spark):
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVER)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .option("maxOffsetsPerTrigger", 500)
        .load()
    )


# Parses raw Kafka bytes into structured columns using wikiEventSchema
def parseToBronze(rawStream):
    return (
        rawStream
        .select(F.from_json(F.col("value").cast(StringType()), wikiEventSchema).alias("e"))
        .select("e.*")
        .filter(F.col("wiki").isNotNull())
    )


# Cleans and enriches Bronze data into Silver layer with derived columns
def bronzeToSilver(bronzeDf):
    return (
        bronzeDf
        .withColumn("eventTime",
            F.to_timestamp(F.from_unixtime(F.col("timestamp"))))
        .withColumn("ingestionTime",
            F.to_timestamp((F.col("ingestion_ts") / 1000).cast("long")))
        .withColumn("latencyMs",
            F.col("ingestion_ts") - (F.col("timestamp") * 1000))
        .withColumn("isAnon",
            F.col("user").rlike(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"))
        .withColumn("editSizeCategory",
            F.when(F.col("byte_delta").isNull(),  "unknown")
            .when(F.col("byte_delta") >  5000,    "large_add")
            .when(F.col("byte_delta") >   500,    "small_add")
            .when(F.col("byte_delta") >=    0,    "tiny")
            .when(F.col("byte_delta") >  -500,    "small_del")
            .otherwise(                           "large_del"))
        .withColumn("commentLen",
            F.length(F.coalesce(F.col("comment"), F.lit(""))))
        .withColumn("hourOfDay",  F.hour(F.col("eventTime")))
        .withColumn("dayOfWeek",  F.dayofweek(F.col("eventTime")))
        .withColumn("wikiLang",
            F.regexp_extract(F.col("wiki"), r"^([a-z]+)", 1))
        .filter(F.col("timestamp") > 0)
        .filter(F.col("type").isin("edit", "new", "categorize", "log"))
    )


# Aggregates Silver data into 5-minute windows for the Gold layer
def silverToGold(silverDf):
    return (
        silverDf
        .filter(F.col("type") == "edit")
        .filter(F.col("namespace") == 0)
        .withWatermark("eventTime", "1 minute")
        .groupBy(F.window("eventTime", "5 minutes"), F.col("wiki"))
        .agg(
            F.count("*")                            .alias("totalEdits"),
            F.sum(F.col("bot").cast("int"))         .alias("botEdits"),
            F.sum(F.col("isAnon").cast("int"))      .alias("anonEdits"),
            F.avg("byte_delta")                     .alias("avgByteDelta"),
            F.approx_count_distinct("user")         .alias("uniqueEditors"),
            F.avg("commentLen")                     .alias("avgCommentLen"),
        )
        .withColumn("botRatio",  F.col("botEdits") / F.col("totalEdits"))
        .withColumn("anonRatio", F.col("anonEdits") / F.col("totalEdits"))
        .withColumn("vandalismFlag",
            (F.col("anonRatio") > 0.3) & (F.col("totalEdits") > 5))
        .withColumn("windowStart", F.col("window.start"))
        .withColumn("windowEnd",   F.col("window.end"))
        .drop("window")
    )


# Writes a streaming DataFrame to Parquet files on disk
def writeToParquet(streamDf, outputPath, checkpointName, outputMode="append"):
    return (
        streamDf.writeStream
        .outputMode(outputMode)
        .format("parquet")
        .option("path", outputPath)
        .option("checkpointLocation", f"{CHECKPOINT_BASE}/{checkpointName}")
        .trigger(processingTime=TRIGGER_TIME)
        .start()
    )


# Writes one row from a Silver batch into Cassandra using a prepared statement
def writeRowToCassandra(session, preparedStmt, row):
    try:
        session.execute(preparedStmt, (
            row["wiki"]          or "unknown",
            row["wikiLang"]      or "unknown",
            row["eventTime"],
            int(row["id"])       if row["id"] else 0,
            int(row["ingestion_ts"]) if row["ingestion_ts"] else 0,
            row["type"]          or "unknown",
            row["title"]         or "",
            row["title_url"]     or "",
            int(row["namespace"]) if row["namespace"] is not None else 0,
            row["user"]          or "anonymous",
            bool(row["bot"])     if row["bot"] is not None else False,
            bool(row["minor"])   if row["minor"] is not None else False,
            bool(row["isAnon"])  if row["isAnon"] is not None else False,
            int(row["byte_delta"]) if row["byte_delta"] is not None else 0,
            int(row["title_len"]) if row["title_len"] is not None else 0,
            int(row["commentLen"]) if row["commentLen"] is not None else 0,
            row["comment"]       or "",
            int(row["hourOfDay"]) if row["hourOfDay"] is not None else 0,
            int(row["dayOfWeek"]) if row["dayOfWeek"] is not None else 1,
            int(row["latencyMs"]) if row["latencyMs"] is not None else 0,
            row["editSizeCategory"] or "unknown",
            row["server_name"]   or "",
        ))
    except Exception as e:
        print("Cassandra write error for row:", str(e)[:100])


# Handles each micro-batch - writes all rows to Cassandra and prints summary
def writeBatchToCassandra(batchDf, batchId):
    if batchDf.isEmpty():
        return

    # Create a fresh Cassandra connection per batch (Spark workers are separate processes)
    cluster = Cluster(
        [CASSANDRA_HOST],
        port=CASSANDRA_PORT,
        load_balancing_policy=RoundRobinPolicy(),
        protocol_version=4,
    )
    session = cluster.connect(CASSANDRA_KS)

    # Prepare the insert statement once per batch for efficiency
    preparedStmt = session.prepare("""
        INSERT INTO wiki_edits (
            wiki, wiki_lang, event_time, event_id, ingestion_ts,
            type, title, title_url, namespace, user_name,
            is_bot, is_minor, is_anon, byte_delta, title_len,
            comment_len, comment, hour_of_day, day_of_week,
            latency_ms, edit_size_cat, server_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """)

    # Collect rows and write to Cassandra
    rows       = batchDf.collect()
    totalRows  = len(rows)
    editCount  = sum(1 for r in rows if r["type"] == "edit")
    botCount   = sum(1 for r in rows if r["bot"] == True)

    for row in rows:
        writeRowToCassandra(session, preparedStmt, row)

    cluster.shutdown()
    print(f"Batch {batchId} | written={totalRows} | edits={editCount} | bots={botCount}")


# Main function - wires up the full pipeline and starts all streaming queries
def main():
    createOutputDirs()
    spark = createSparkSession()

    print("Pipeline starting - Kafka topic:", KAFKA_TOPIC)
    print("Cassandra host:", CASSANDRA_HOST)

    # Build the three layers
    rawStream = readKafkaStream(spark)
    bronzeDf  = parseToBronze(rawStream)
    silverDf  = bronzeToSilver(bronzeDf)
    goldDf    = silverToGold(silverDf)

    # Write Bronze to parquet
    bronzeQuery = writeToParquet(bronzeDf, BRONZE_PATH, "bronze")
    print("Bronze writing to:", BRONZE_PATH)

    # Write Silver to parquet
    silverQuery = writeToParquet(silverDf, SILVER_PATH, "silver")
    print("Silver writing to:", SILVER_PATH)

    # Write Gold to parquet
    goldQuery = writeToParquet(goldDf, GOLD_PATH, "gold")
    print("Gold writing to:", GOLD_PATH)

    # Write Silver to Cassandra live (foreachBatch)
    cassandraQuery = (
        silverDf.writeStream
        .foreachBatch(writeBatchToCassandra)
        .option("checkpointLocation", f"{CHECKPOINT_BASE}/cassandra")
        .trigger(processingTime=TRIGGER_TIME)
        .start()
    )
    print("Cassandra live write started")

    print("All queries running. Press Ctrl+C to stop.")

    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        print("Stopping pipeline...")
        for query in spark.streams.active:
            query.stop()
        spark.stop()
        print("Pipeline stopped.")


if __name__ == "__main__":
    main()