# setup_cassandra.py
# Creates the keyspace and table in Cassandra for Wikipedia edit data
#
# Run once before starting the streaming pipeline:
#   python3 setup_cassandra.py

from cassandra.cluster import Cluster
from cassandra.policies import RoundRobinPolicy

CASSANDRA_HOST = "127.0.0.1"
CASSANDRA_PORT = 9042
KEYSPACE       = "wikipedia"


# Connects to Cassandra and returns cluster and session objects
def connectToCassandra():
    cluster = Cluster(
        [CASSANDRA_HOST],
        port=CASSANDRA_PORT,
        load_balancing_policy=RoundRobinPolicy(),
        protocol_version=4,
    )
    session = cluster.connect()
    print("Connected to Cassandra at", CASSANDRA_HOST)
    return cluster, session


# Creates the wikipedia keyspace with simple replication factor 1
def createKeyspace(session):
    session.execute(f"""
        CREATE KEYSPACE IF NOT EXISTS {KEYSPACE}
        WITH replication = {{
            'class': 'SimpleStrategy',
            'replication_factor': 1
        }}
    """)
    session.set_keyspace(KEYSPACE)
    print("Keyspace ready:", KEYSPACE)


# Creates the wiki_edits table to store all Silver layer streaming data
# Partition key is (wiki, wiki_lang) so queries per wiki are efficient
def createEditsTable(session):
    session.execute("""
        CREATE TABLE IF NOT EXISTS wiki_edits (
            wiki          TEXT,
            wiki_lang     TEXT,
            event_time    TIMESTAMP,
            event_id      BIGINT,
            ingestion_ts  BIGINT,
            type          TEXT,
            title         TEXT,
            title_url     TEXT,
            namespace     INT,
            user_name     TEXT,
            is_bot        BOOLEAN,
            is_minor      BOOLEAN,
            is_anon       BOOLEAN,
            byte_delta    INT,
            title_len     INT,
            comment_len   INT,
            comment       TEXT,
            hour_of_day   INT,
            day_of_week   INT,
            latency_ms    BIGINT,
            edit_size_cat TEXT,
            server_name   TEXT,
            PRIMARY KEY ((wiki, wiki_lang), event_time, event_id)
        ) WITH CLUSTERING ORDER BY (event_time DESC, event_id DESC)
          AND default_time_to_live = 604800
    """)
    print("Table ready: wiki_edits")


# Creates a summary table for fast aggregation queries
def createSummaryTable(session):
    session.execute("""
        CREATE TABLE IF NOT EXISTS wiki_hourly_summary (
            wiki          TEXT,
            wiki_lang     TEXT,
            hour_bucket   TIMESTAMP,
            total_edits   COUNTER,
            PRIMARY KEY ((wiki, wiki_lang), hour_bucket)
        ) WITH CLUSTERING ORDER BY (hour_bucket DESC)
    """)
    print("Table ready: wiki_hourly_summary")


# Verifies setup by listing tables in the wikipedia keyspace
def verifySetup(session):
    rows   = session.execute("SELECT table_name FROM system_schema.tables WHERE keyspace_name = 'wikipedia'")
    tables = [row.table_name for row in rows]
    print("Tables in wikipedia keyspace:", tables)


# Main function - runs full Cassandra setup in order
def main():
    cluster, session = connectToCassandra()
    createKeyspace(session)
    createEditsTable(session)
    createSummaryTable(session)
    verifySetup(session)
    cluster.shutdown()
    print("\nCassandra setup complete. Ready for streaming.")


if __name__ == "__main__":
    main()