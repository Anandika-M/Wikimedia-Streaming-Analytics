# schema.py
# Defines the Spark schema that matches the raw Wikimedia Kafka messages
# Based on actual message structure from the wikipedia Kafka topic

from pyspark.sql.types import (
    BooleanType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# Schema for the nested "meta" object inside each event
metaSchema = StructType([
    StructField("uri",        StringType(),  True),
    StructField("request_id", StringType(),  True),
    StructField("id",         StringType(),  True),
    StructField("dt",         StringType(),  True),
    StructField("domain",     StringType(),  True),
    StructField("stream",     StringType(),  True),
    StructField("topic",      StringType(),  True),
    StructField("partition",  IntegerType(), True),
    StructField("offset",     LongType(),    True),
])

# Schema for the nested "length" object (old and new byte size of the page)
lengthSchema = StructType([
    StructField("old", IntegerType(), True),
    StructField("new", IntegerType(), True),
])

# Schema for the nested "revision" object (old and new revision IDs)
revisionSchema = StructType([
    StructField("old", LongType(), True),
    StructField("new", LongType(), True),
])

# Top-level schema for the full Wikimedia recentchange event
# Covers all event types: edit, new, categorize, log
wikiEventSchema = StructType([

    # Standard fields present in all event types
    StructField("schema",             StringType(),  True),  # mediawiki schema version
    StructField("id",                 LongType(),    True),  # unique event ID
    StructField("type",               StringType(),  True),  # edit, new, categorize, log
    StructField("wiki",               StringType(),  True),  # e.g. enwiki, commonswiki
    StructField("title",              StringType(),  True),  # page title
    StructField("title_url",          StringType(),  True),  # full URL to the page
    StructField("namespace",          IntegerType(), True),  # 0=article, 14=category, etc
    StructField("user",               StringType(),  True),  # editor username or IP
    StructField("bot",                BooleanType(), True),  # true if made by a bot
    StructField("minor",              BooleanType(), True),  # true if marked as minor edit
    StructField("patrolled",          BooleanType(), True),  # true if edit was patrolled
    StructField("timestamp",          LongType(),    True),  # unix epoch seconds
    StructField("comment",            StringType(),  True),  # raw edit summary
    StructField("parsedcomment",      StringType(),  True),  # HTML version of edit summary
    StructField("notify_url",         StringType(),  True),  # diff URL
    StructField("server_url",         StringType(),  True),  # e.g. https://en.wikipedia.org
    StructField("server_name",        StringType(),  True),  # e.g. en.wikipedia.org
    StructField("server_script_path", StringType(),  True),  # usually /w

    # Nested objects
    StructField("meta",     metaSchema,     True),  # event metadata
    StructField("length",   lengthSchema,   True),  # page size before and after edit
    StructField("revision", revisionSchema, True),  # revision IDs before and after edit

    # Log-specific fields (only present when type = log)
    StructField("log_id",             LongType(),   True),
    StructField("log_type",           StringType(), True),  # e.g. upload, delete, move
    StructField("log_action",         StringType(), True),  # e.g. upload, restore
    StructField("log_action_comment", StringType(), True),

    # Fields added by the producer (wikimedia_kafka_producer.py enrichment)
    StructField("ingestion_ts",  LongType(),    True),  # epoch ms when producer received it
    StructField("is_bot",        BooleanType(), True),  # copy of bot field
    StructField("namespace_str", StringType(),  True),  # human label e.g. article, file
    StructField("byte_delta",    IntegerType(), True),  # length.new - length.old
    StructField("title_len",     IntegerType(), True),  # character count of title
])