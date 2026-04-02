"""
=============================================================
  Wikimedia EventStream → Kafka Producer
  Topic : wikipedia
  Stream: https://stream.wikimedia.org/v2/stream/recentchange
=============================================================
  Install dependencies:
      pip install kafka-python sseclient-py requests

  Run:
      python wikimedia_kafka_producer.py
"""

import json
import logging
import signal
import sys
import time
from datetime import datetime

import requests
import sseclient
from kafka import KafkaProducer
from kafka.errors import KafkaError

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = ["localhost:9092"]   # change if remote broker
KAFKA_TOPIC              = "wikipedia"

WIKIMEDIA_SSE_URL = "https://stream.wikimedia.org/v2/stream/recentchange"

# ── Wikimedia requires a descriptive User-Agent or it returns 403 ──────────
# Format: <tool-name>/<version> (<contact-or-repo>) <runtime>
# See: https://meta.wikimedia.org/wiki/User-Agent_policy
HEADERS = {
    "User-Agent": (
        "WikimediaKafkaProducer/1.0 "
        "(bigdata-project; https://github.com/your-repo) "
        "python-requests/2.x"
    ),
    "Accept": "text/event-stream",
}

# Optional filters – set to None to ingest everything
FILTER_WIKI   = None          # e.g. "enwiki" → English Wikipedia only
FILTER_TYPES  = None          # e.g. ["edit", "new"] → edits & new pages only

# Throughput controls
BATCH_LOG_EVERY  = 500        # log a stats line every N messages
REQUEST_TIMEOUT  = 60         # seconds for SSE HTTP connection

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wikimedia-producer")


# ─────────────────────────────────────────────
#  Serialiser  (dict → bytes)
# ─────────────────────────────────────────────
def json_serializer(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


# ─────────────────────────────────────────────
#  Kafka key extractor
#  Partition by wiki name so each project's
#  edits land on the same partition (ordering).
# ─────────────────────────────────────────────
def extract_key(event: dict) -> bytes:
    wiki = event.get("wiki", "unknown")
    return wiki.encode("utf-8")


# ─────────────────────────────────────────────
#  Enrich / flatten the raw SSE payload
# ─────────────────────────────────────────────
def transform_event(raw: dict) -> dict:
    """
    Keep all original fields and add derived columns useful for
    downstream Spark ML:
      ingestion_ts  – epoch ms when the producer received this event
      is_bot        – boolean flag from meta
      namespace_str – human-readable namespace label
      byte_delta    – size change (new_len - old_len), None for new pages
      title_len     – character length of the page title
    """
    namespace_map = {
        0: "article", 1: "talk", 2: "user", 3: "user_talk",
        4: "wikipedia", 6: "file", 10: "template", 14: "category",
    }

    ns   = raw.get("namespace", -1)
    nlen = raw.get("length", {}).get("new")
    olen = raw.get("length", {}).get("old")

    enriched = {
        **raw,                                          # all original fields
        "ingestion_ts"  : int(datetime.utcnow().timestamp() * 1000),
        "is_bot"        : raw.get("bot", False),
        "namespace_str" : namespace_map.get(ns, f"ns_{ns}"),
        "byte_delta"    : (nlen - olen) if (nlen is not None and olen is not None) else None,
        "title_len"     : len(raw.get("title", "")),
    }
    return enriched


# ─────────────────────────────────────────────
#  Delivery callback
# ─────────────────────────────────────────────
def on_send_success(record_metadata):
    pass   # uncomment below for per-message logging (very verbose)
    # log.debug("topic=%s partition=%d offset=%d",
    #            record_metadata.topic, record_metadata.partition,
    #            record_metadata.offset)


def on_send_error(exc):
    log.error("Kafka delivery error: %s", exc)


# ─────────────────────────────────────────────
#  Graceful shutdown
# ─────────────────────────────────────────────
_running = True

def _shutdown(sig, frame):
    global _running
    log.info("Shutdown signal received – draining producer …")
    _running = False

signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ─────────────────────────────────────────────
#  Main producer loop
# ─────────────────────────────────────────────
def main():
    log.info("Connecting to Kafka at %s …", KAFKA_BOOTSTRAP_SERVERS)

    producer = KafkaProducer(
        bootstrap_servers      = KAFKA_BOOTSTRAP_SERVERS,
        value_serializer       = json_serializer,
        key_serializer         = lambda k: k,          # already bytes
        acks                   = "all",                 # strongest durability
        retries                = 5,
        max_in_flight_requests_per_connection = 1,      # preserve ordering
        compression_type       = "gzip",                # JSON compresses well
        linger_ms              = 20,                    # small batching window
        batch_size             = 32_768,                # 32 KB batch
    )
    log.info("Kafka producer ready.")

    total       = 0
    filtered_out = 0
    errors      = 0
    start_time  = time.time()

    while _running:
        try:
            log.info("Opening SSE connection to Wikimedia …")
            response = requests.get(
                WIKIMEDIA_SSE_URL,
                stream=True,
                timeout=REQUEST_TIMEOUT,
                headers=HEADERS,
            )
            response.raise_for_status()

            client = sseclient.SSEClient(response)

            for sse_event in client.events():
                if not _running:
                    break

                # SSE heartbeats arrive as empty data – skip
                if not sse_event.data or sse_event.data.strip() == "":
                    continue

                try:
                    raw = json.loads(sse_event.data)
                except json.JSONDecodeError as e:
                    log.warning("JSON parse error: %s | data=%s", e, sse_event.data[:120])
                    errors += 1
                    continue

                # ── Optional filters ──────────────────────
                if FILTER_WIKI and raw.get("wiki") != FILTER_WIKI:
                    filtered_out += 1
                    continue

                if FILTER_TYPES and raw.get("type") not in FILTER_TYPES:
                    filtered_out += 1
                    continue
                # ─────────────────────────────────────────

                event = transform_event(raw)
                key   = extract_key(event)

                (
                    producer
                    .send(KAFKA_TOPIC, key=key, value=event)
                    .add_callback(on_send_success)
                    .add_errback(on_send_error)
                )

                total += 1

                if total % BATCH_LOG_EVERY == 0:
                    elapsed = time.time() - start_time
                    rate    = total / elapsed if elapsed > 0 else 0
                    log.info(
                        "Published %6d msgs | rate=%.1f msg/s | "
                        "filtered=%d | errors=%d",
                        total, rate, filtered_out, errors
                    )

        except requests.exceptions.RequestException as e:
            log.error("SSE connection error: %s – retrying in 5 s …", e)
            time.sleep(5)

        except Exception as e:
            log.exception("Unexpected error: %s – retrying in 5 s …", e)
            time.sleep(5)

    # ── Flush & close ────────────────────────
    log.info("Flushing producer (max 30 s) …")
    producer.flush(timeout=30)
    producer.close()

    elapsed = time.time() - start_time
    log.info(
        "Producer stopped. total=%d  filtered=%d  errors=%d  "
        "elapsed=%.1fs  avg_rate=%.1f msg/s",
        total, filtered_out, errors, elapsed,
        total / elapsed if elapsed > 0 else 0,
    )


if __name__ == "__main__":
    main()
