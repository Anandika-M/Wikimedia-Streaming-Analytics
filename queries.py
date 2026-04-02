# queries.py
# 12 static analytical queries against the Cassandra wikipedia keyspace
#
# Run standalone:
#   python3 queries.py

import pandas as pd
from cassandra.cluster import Cluster
from cassandra.policies import RoundRobinPolicy

CASSANDRA_HOST = "127.0.0.1"
CASSANDRA_PORT = 9042
KEYSPACE       = "wikipedia"


# Creates and returns a Cassandra session connected to the wikipedia keyspace
def getSession():
    cluster = Cluster(
        [CASSANDRA_HOST],
        port=CASSANDRA_PORT,
        load_balancing_policy=RoundRobinPolicy(),
        protocol_version=4,
    )
    session = cluster.connect(KEYSPACE)
    return cluster, session


# Fetches all rows from wiki_edits and returns as a pandas DataFrame
def fetchAllEdits(session):
    rows = session.execute("SELECT * FROM wiki_edits")
    df   = pd.DataFrame(list(rows))
    if df.empty:
        return df
    df.columns = [c.lower() for c in df.columns]
    return df


# Q1 - Edit Spike Detection
# Pages with edit counts above 2x the average - potential trending topics
def queryEditSpikes(df):
    if df.empty or "title" not in df.columns:
        return pd.DataFrame()
    editOnly   = df[df["type"] == "edit"].copy()
    pageCounts = editOnly.groupby("title").size().reset_index(name="editCount")
    avgCount   = pageCounts["editCount"].mean()
    spikes     = pageCounts[pageCounts["editCount"] > 2 * avgCount]
    return spikes.sort_values("editCount", ascending=False).head(20)


# Q2 - Median Edit Size per Wiki
# Median byte_delta per wiki - more robust than average
def queryMedianEditSize(df):
    if df.empty or "byte_delta" not in df.columns:
        return pd.DataFrame()
    editOnly = df[df["type"] == "edit"].copy()
    return (
        editOnly.groupby("wiki")["byte_delta"]
        .median().reset_index()
        .rename(columns={"byte_delta": "medianByteDelta"})
        .sort_values("medianByteDelta", ascending=False).head(20)
    )


# Q3 - Multi-User Rapid Edit Pages
# Pages edited by 3+ distinct users within a 1-hour window
def queryMultiUserPages(df):
    if df.empty or "title" not in df.columns:
        return pd.DataFrame()
    editOnly = df[df["type"] == "edit"].copy()
    editOnly["event_time"] = pd.to_datetime(editOnly["event_time"], errors="coerce")
    editOnly["hourBucket"] = editOnly["event_time"].dt.floor("1H")
    grouped = (
        editOnly.groupby(["title", "hourBucket"])
        .agg(uniqueUsers=("user_name", "nunique"), totalEdits=("title", "count"))
        .reset_index()
    )
    return (
        grouped[grouped["uniqueUsers"] >= 3]
        .sort_values("uniqueUsers", ascending=False).head(20)
    )


# Q4 - Repeated Edits by Same User on Same Page
# Users editing the same page 3+ times - possible edit wars
def queryRepeatedEdits(df):
    if df.empty or "user_name" not in df.columns:
        return pd.DataFrame()
    editOnly = df[df["type"] == "edit"].copy()
    grouped  = editOnly.groupby(["user_name", "title"]).size().reset_index(name="editCount")
    return grouped[grouped["editCount"] >= 3].sort_values("editCount", ascending=False).head(20)


# Q5 - Cross-Language Contributors
# Users contributing across multiple wiki languages
def queryCrossLanguageContributors(df):
    if df.empty or "wiki_lang" not in df.columns:
        return pd.DataFrame()
    result = (
        df.groupby("user_name")["wiki_lang"]
        .nunique().reset_index(name="languageCount")
    )
    return result[result["languageCount"] >= 2].sort_values("languageCount", ascending=False).head(20)


# Q6 - Edit Session Identification
# Groups edits into sessions (same user, gap <= 30 min between edits)
def queryEditSessions(df):
    if df.empty or "user_name" not in df.columns:
        return pd.DataFrame()
    editOnly = df[df["type"] == "edit"].copy()
    editOnly["event_time"] = pd.to_datetime(editOnly["event_time"], errors="coerce")
    editOnly = editOnly.sort_values(["user_name", "event_time"])
    editOnly["timeDiff"]   = editOnly.groupby("user_name")["event_time"].diff()
    editOnly["newSession"] = (editOnly["timeDiff"] > pd.Timedelta(minutes=30)) | editOnly["timeDiff"].isna()
    editOnly["sessionId"]  = editOnly.groupby("user_name")["newSession"].cumsum()
    sessions = (
        editOnly.groupby(["user_name", "sessionId"])
        .agg(sessionEdits=("title", "count"), sessionStart=("event_time", "min"), sessionEnd=("event_time", "max"))
        .reset_index()
    )
    sessions["durationMin"] = (sessions["sessionEnd"] - sessions["sessionStart"]).dt.total_seconds() / 60
    return sessions.sort_values("sessionEdits", ascending=False).head(20)


# Q7 - Longest Editing Streak
# Maximum number of edits a user makes in a single day
def queryLongestEditingStreak(df):
    if df.empty or "user_name" not in df.columns:
        return pd.DataFrame()
    editOnly = df[df["type"] == "edit"].copy()
    editOnly["event_time"] = pd.to_datetime(editOnly["event_time"], errors="coerce")
    editOnly["date"]       = editOnly["event_time"].dt.date
    return (
        editOnly.groupby(["user_name", "date"]).size()
        .reset_index(name="editsInDay")
        .sort_values("editsInDay", ascending=False).head(20)
    )


# Q8 - High-Impact Contributors
# Non-bot users ranked by average absolute byte_delta per edit
def queryHighImpactContributors(df):
    if df.empty or "byte_delta" not in df.columns:
        return pd.DataFrame()
    editOnly = df[(df["type"] == "edit") & (df["is_bot"] == False)].copy()
    editOnly["absDelta"] = editOnly["byte_delta"].abs()
    return (
        editOnly.groupby("user_name")
        .agg(totalEdits=("title", "count"), avgAbsDelta=("absDelta", "mean"), totalBytes=("byte_delta", "sum"))
        .query("totalEdits >= 2").round(2).reset_index()
        .sort_values("avgAbsDelta", ascending=False).head(20)
    )


# Q9 - Comment Length vs Edit Size Correlation
# Does longer comment length correlate with larger edits?
def queryCommentVsEditSize(df):
    if df.empty or "comment_len" not in df.columns:
        return pd.DataFrame()
    editOnly = df[df["type"] == "edit"].copy()
    editOnly["absDelta"]   = editOnly["byte_delta"].abs()
    editOnly["commentBin"] = pd.cut(
        editOnly["comment_len"],
        bins=[0, 10, 30, 60, 100, 500],
        labels=["0-10", "10-30", "30-60", "60-100", "100+"]
    )
    return (
        editOnly.groupby("commentBin", observed=True)
        .agg(count=("title", "count"), avgByteDelta=("byte_delta", "mean"), avgAbsDelta=("absDelta", "mean"))
        .round(2).reset_index()
    )


# Q10 - Edit Size Distribution per Wiki
# Breakdown of edit_size_cat across different wikis
def queryEditSizeDistribution(df):
    if df.empty or "edit_size_cat" not in df.columns:
        return pd.DataFrame()
    editOnly = df[df["type"] == "edit"].copy()
    pivot    = (
        editOnly.groupby(["wiki_lang", "edit_size_cat"])
        .size().unstack(fill_value=0).reset_index()
    )
    return pivot.head(15)


# Q11 - Collaboration Clusters
# Pages with 3+ distinct editors - co-editing patterns
def queryCollaborationClusters(df):
    if df.empty or "user_name" not in df.columns:
        return pd.DataFrame()
    editOnly = df[df["type"] == "edit"].copy()
    coEdits  = editOnly.groupby(["title", "user_name"]).size().reset_index(name="editCount")
    return (
        coEdits.groupby("title")["user_name"].nunique()
        .reset_index(name="uniqueEditors")
        .query("uniqueEditors >= 3")
        .sort_values("uniqueEditors", ascending=False).head(20)
    )


# Q12 - Outlier Edit Detection
# Edits with byte_delta above the 95th percentile
def queryOutlierEdits(df):
    if df.empty or "byte_delta" not in df.columns:
        return pd.DataFrame()
    editOnly  = df[df["type"] == "edit"].copy()
    threshold = editOnly["byte_delta"].abs().quantile(0.95)
    outliers  = editOnly[editOnly["byte_delta"].abs() > threshold].copy()
    return (
        outliers[["wiki", "title", "user_name", "byte_delta", "comment_len", "is_bot", "is_anon", "event_time"]]
        .sort_values("byte_delta", ascending=False).head(20)
    )


# Registry of all 12 queries used by app.py to build the dropdown
QUERY_REGISTRY = [
    {"key": "q1",  "label": "Q1  - Edit Spike Detection",       "desc": "Pages with edits above 2x average — trending or breaking news topics.",       "fn": queryEditSpikes,                "xCol": "title",       "yCol": "editCount"},
    {"key": "q2",  "label": "Q2  - Median Edit Size per Wiki",  "desc": "Median byte_delta per wiki — robust measure of typical edit size.",           "fn": queryMedianEditSize,            "xCol": "wiki",        "yCol": "medianByteDelta"},
    {"key": "q3",  "label": "Q3  - Multi-User Rapid Edit Pages","desc": "Pages edited by 3+ users within 1 hour — collaboration or conflict.",         "fn": queryMultiUserPages,            "xCol": "title",       "yCol": "uniqueUsers"},
    {"key": "q4",  "label": "Q4  - Repeated Edits by Same User","desc": "Users editing the same page 3+ times — edit wars or corrections.",            "fn": queryRepeatedEdits,             "xCol": "user_name",   "yCol": "editCount"},
    {"key": "q5",  "label": "Q5  - Cross-Language Contributors","desc": "Users contributing across 2+ wiki languages — global contributors.",          "fn": queryCrossLanguageContributors, "xCol": "user_name",   "yCol": "languageCount"},
    {"key": "q6",  "label": "Q6  - Edit Session Identification","desc": "Edits grouped into sessions (gap <= 30 min) — user behaviour patterns.",     "fn": queryEditSessions,              "xCol": "user_name",   "yCol": "sessionEdits"},
    {"key": "q7",  "label": "Q7  - Longest Editing Streak",     "desc": "Max edits by a user in a single day — engagement metric.",                   "fn": queryLongestEditingStreak,      "xCol": "user_name",   "yCol": "editsInDay"},
    {"key": "q8",  "label": "Q8  - High-Impact Contributors",   "desc": "Non-bot users ranked by avg absolute byte change — quality over quantity.",   "fn": queryHighImpactContributors,    "xCol": "user_name",   "yCol": "avgAbsDelta"},
    {"key": "q9",  "label": "Q9  - Comment Length vs Edit Size","desc": "Does longer comment correlate with larger edits? Grouped by length bin.",     "fn": queryCommentVsEditSize,         "xCol": "commentBin",  "yCol": "avgByteDelta"},
    {"key": "q10", "label": "Q10 - Edit Size Distribution",     "desc": "Breakdown of edit size categories across different wikis.",                   "fn": queryEditSizeDistribution,      "xCol": None,          "yCol": None},
    {"key": "q11", "label": "Q11 - Collaboration Clusters",     "desc": "Pages with 3+ distinct editors — co-editing and network patterns.",           "fn": queryCollaborationClusters,     "xCol": "title",       "yCol": "uniqueEditors"},
    {"key": "q12", "label": "Q12 - Outlier Edit Detection",     "desc": "Edits above the 95th percentile — possible vandalism or major updates.",     "fn": queryOutlierEdits,              "xCol": "title",       "yCol": "byte_delta"},
]


# Runs all 12 queries standalone and prints results to console
def runAllQueries():
    print("Connecting to Cassandra...")
    cluster, session = getSession()
    print("Fetching data...")
    df = fetchAllEdits(session)
    if df.empty:
        print("No data found. Start the streaming pipeline first.")
        cluster.shutdown()
        return
    print(f"Loaded {len(df)} rows.\n")
    for q in QUERY_REGISTRY:
        print(f"\n{'='*55}\n  {q['label']}\n{'='*55}")
        result = q["fn"](df)
        print(result.to_string(index=False) if not result.empty else "  No results.")
    cluster.shutdown()


if __name__ == "__main__":
    runAllQueries()