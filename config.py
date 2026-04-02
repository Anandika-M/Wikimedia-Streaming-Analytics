"""
=============================================================
  config.py  —  Central configuration for all pipeline steps
=============================================================
"""

# ── Kafka ─────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC             = "wikipedia"

# ── Spark ─────────────────────────────────────────────────
SPARK_APP_NAME          = "WikipediaPipeline"
SPARK_MASTER            = "local[*]"          # use all local CPU cores

# Kafka-Spark connector JAR (Spark 3.5 + Kafka 2.x/3.x)
SPARK_KAFKA_PACKAGE     = (
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"
)

# ── Streaming ─────────────────────────────────────────────
TRIGGER_INTERVAL        = "10 seconds"        # micro-batch interval
CHECKPOINT_BASE         = "/tmp/wiki_checkpoints"

# ── Output paths (local filesystem) ───────────────────────
OUTPUT_BASE             = "/tmp/wiki_output"
BRONZE_PATH             = f"{OUTPUT_BASE}/bronze"   # raw JSON
SILVER_PATH             = f"{OUTPUT_BASE}/silver"   # cleaned + typed
GOLD_PATH               = f"{OUTPUT_BASE}/gold"     # aggregated
MODEL_PATH              = f"{OUTPUT_BASE}/models"   # saved ML models

# ── ML ─────────────────────────────────────────────────────
RANDOM_SEED             = 42
TEST_SPLIT              = 0.2
