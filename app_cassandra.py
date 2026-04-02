# app_cassandra.py
# Streamlit dashboard showing all 15 static query results from Cassandra
#
# Run with:
#   streamlit run app_cassandra.py

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from cassandra.cluster import Cluster
from cassandra.policies import RoundRobinPolicy

from queries import (
    fetchAllEdits,
    queryEditSpikes,
    queryNegativeLatency,
    queryMedianEditSize,
    queryMultiUserPages,
    queryRepeatedEdits,
    queryCrossLanguageContributors,
    queryEditSessions,
    queryLongestEditingStreak,
    queryHighImpactContributors,
    queryCommentVsEditSize,
    queryEditSizeDistribution,
    queryDominantEditors,
    queryCollaborationClusters,
    queryOutlierEdits,
    queryTimeBehaviorProfiling,
)

CASSANDRA_HOST = "127.0.0.1"
CASSANDRA_PORT = 9042
KEYSPACE       = "wikipedia"

st.set_page_config(
    page_title="Wikipedia Cassandra Analytics",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .main { background-color: #0a0e1a; }
    .block-container { padding-top: 1.5rem; }

    .query-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .query-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #6366f1;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .query-desc {
        font-size: 0.9rem;
        color: #9ca3af;
        margin-bottom: 0;
    }
    .metric-pill {
        display: inline-block;
        background: #1e1b4b;
        color: #818cf8;
        border: 1px solid #3730a3;
        border-radius: 20px;
        padding: 3px 14px;
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 2px;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
        display: block;
    }
    div[data-testid="stSidebar"] { background: #070b14; border-right: 1px solid #1f2937; }
    h1, h2, h3 { color: #e2e8f0 !important; }
    .stTabs [data-baseweb="tab"] { color: #6b7280; font-size: 0.85rem; }
    .stTabs [aria-selected="true"] { color: #6366f1; border-bottom-color: #6366f1; }
    .stDataFrame { border: 1px solid #1f2937; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# Connects to Cassandra and returns cluster and session
def connectCassandra():
    cluster = Cluster(
        [CASSANDRA_HOST],
        port=CASSANDRA_PORT,
        load_balancing_policy=RoundRobinPolicy(),
        protocol_version=4,
    )
    session = cluster.connect(KEYSPACE)
    return cluster, session


# Loads all data from Cassandra with caching (refreshes every 60 seconds)
@st.cache_data(ttl=60)
def loadData():
    try:
        cluster, session = connectCassandra()
        df = fetchAllEdits(session)
        cluster.shutdown()
        return df
    except Exception as e:
        return pd.DataFrame()


# Renders a query card header with title, description and result count
def queryCard(title, description, resultCount=None):
    countHtml = f'<span class="metric-pill">{resultCount}</span> rows' if resultCount is not None else ""
    st.markdown(f"""
    <div class="query-card">
        <p class="query-title">{title}</p>
        <p class="query-desc">{description} {countHtml}</p>
    </div>
    """, unsafe_allow_html=True)


# Renders a small bar chart for a query result DataFrame
def quickBarChart(df, xCol, yCol, title, color="#6366f1"):
    if df.empty or xCol not in df.columns or yCol not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    plotDf = df.head(15)
    ax.barh(plotDf[xCol].astype(str).str[:35], plotDf[yCol], color=color)
    ax.set_title(title, color="#e2e8f0", fontsize=11, pad=8)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1f2937")
    ax.spines["bottom"].set_color("#1f2937")
    ax.invert_yaxis()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# Renders the sidebar navigation and connection status
def renderSidebar(df):
    with st.sidebar:
        st.markdown("## Cassandra\nWikipedia Analytics")
        st.markdown("---")

        if df.empty:
            st.error("No data in Cassandra")
            
            st.success(f"{len(df):,} rows loaded")
            if "wiki" in df.columns:
                st.markdown(f"**Wikis:** {df['wiki'].nunique()}")
            if "user_name" in df.columns:
                st.markdown(f"**Users:** {df['user_name'].nunique()}")
            if "title" in df.columns:
                st.markdown(f"**Pages:** {df['title'].nunique()}")

        st.markdown("---")
        st.markdown("**15 Queries**")
        queries = [
            "Q1 Edit Spike Detection",
            "Q2 Median Edit Size",
            "Q3 Multi-User Pages",
            "Q4 Repeated Edits",
            "Q5 Cross-Language",
            "Q6 Edit Sessions",
            "Q7 Editing Streak",
            "Q8 High-Impact Users",
            "Q9 Comment vs Size",
            "Q10 Size Distribution",
            "Q11 Collaboration",
            "Q12 Outlier Edits"
        ]
        for q in queries:
            st.caption(q)

        st.markdown("---")
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()


# Renders the overview tab with top-level stats and data sample
def renderOverview(df):
    st.markdown("## Wikipedia Live Edit Analytics — Cassandra Backend")
    st.markdown("Real-time Wikipedia edit data streamed from Kafka through Spark into Cassandra.")
    st.markdown("---")

    if df.empty:
        st.warning("No data found in Cassandra. Start the streaming pipeline first.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<span class="metric-pill">{len(df):,}</span><span class="metric-label">Total Events</span>', unsafe_allow_html=True)
    with col2:
        editCount = len(df[df["type"] == "edit"]) if "type" in df.columns else 0
        st.markdown(f'<span class="metric-pill">{editCount:,}</span><span class="metric-label">Article Edits</span>', unsafe_allow_html=True)
    with col3:
        wikis = df["wiki"].nunique() if "wiki" in df.columns else 0
        st.markdown(f'<span class="metric-pill">{wikis}</span><span class="metric-label">Unique Wikis</span>', unsafe_allow_html=True)
    with col4:
        users = df["user_name"].nunique() if "user_name" in df.columns else 0
        st.markdown(f'<span class="metric-pill">{users}</span><span class="metric-label">Unique Users</span>', unsafe_allow_html=True)

    st.markdown("### Sample Data from Cassandra")
    displayCols = ["wiki", "type", "title", "user_name", "is_bot", "byte_delta", "comment_len", "hour_of_day", "edit_size_cat"]
    availCols   = [c for c in displayCols if c in df.columns]
    st.dataframe(df[availCols].head(50), use_container_width=True)


# Renders a single query tab with results table and chart
def renderQueryTab(label, description, resultDf, xCol=None, yCol=None, chartTitle=""):
    queryCard(label, description, len(resultDf) if not resultDf.empty else 0)
    if resultDf.empty:
        st.info("No results found for this query. Collect more streaming data and refresh.")
        return
    col1, col2 = st.columns([3, 2])
    with col1:
        st.dataframe(resultDf, use_container_width=True)
    with col2:
        if xCol and yCol and xCol in resultDf.columns and yCol in resultDf.columns:
            quickBarChart(resultDf, xCol, yCol, chartTitle)
        else:
            st.caption("No chart available for this query.")


# Main app function - loads data and renders all tabs
def main():
    df = loadData()
    renderSidebar(df)

    tabs = st.tabs([
        "Overview",
        "Q1 Spikes", "Q2 Latency", "Q3 Median Size",
        "Q4 Multi-User", "Q5 Repeated", "Q6 Cross-Lang",
        "Q7 Sessions", "Q8 Streak", "Q9 Impact",
        "Q10 Comments", "Q11 Size Dist", "Q12 Dominant",
        "Q13 Collab", "Q14 Outliers", "Q15 Time",
    ])

    with tabs[0]:
        renderOverview(df)

    with tabs[1]:
        renderQueryTab(
            "Q1 - Edit Spike Detection",
            "Pages with edit counts above 2x the average — potential trending or breaking news topics.",
            queryEditSpikes(df), "title", "editCount", "Edit Count per Page",
        )
    with tabs[2]:
        renderQueryTab(
            "Q2 - Median Edit Size per Wiki",
            "Median byte_delta per wiki — more robust than average for skewed distributions.",
            queryMedianEditSize(df), "wiki", "medianByteDelta", "Median Byte Delta by Wiki",
        )
    with tabs[3]:
        renderQueryTab(
            "Q3 - Multi-User Rapid Edit Pages",
            "Pages edited by 3+ distinct users within a 1-hour window — collaboration or conflict.",
            queryMultiUserPages(df), "title", "uniqueUsers", "Unique Users per Page per Hour",
        )
    with tabs[4]:
        renderQueryTab(
            "Q4 - Repeated Edits by Same User",
            "Users editing the same page 3+ times — possible edit wars or iterative corrections.",
            queryRepeatedEdits(df), "user_name", "editCount", "Repeated Edits by User",
        )
    with tabs[5]:
        renderQueryTab(
            "Q5 - Cross-Language Contributors",
            "Users contributing across 2+ wiki languages — high-value global contributors.",
            queryCrossLanguageContributors(df), "user_name", "languageCount", "Languages per User",
        )
    with tabs[6]:
        renderQueryTab(
            "Q6 - Edit Session Identification",
            "Edits grouped into sessions (same user, gap <= 30 min) — user behaviour patterns.",
            queryEditSessions(df), "user_name", "sessionEdits", "Edits per Session",
        )
    with tabs[7]:
        renderQueryTab(
            "Q7 - Longest Editing Streak",
            "Maximum edits by a user in a single day — engagement and activity metric.",
            queryLongestEditingStreak(df), "user_name", "editsInDay", "Edits in a Single Day",
        )
    with tabs[8]:
        renderQueryTab(
            "Q8 - High-Impact Contributors",
            "Non-bot users ranked by average absolute byte change — quality over quantity.",
            queryHighImpactContributors(df), "user_name", "avgAbsDelta", "Avg Absolute Delta per User",
        )
    with tabs[9]:
        renderQueryTab(
            "Q9 - Comment Length vs Edit Size",
            "Does longer comment correlate with larger edits? Grouped by comment length bin.",
            queryCommentVsEditSize(df), "commentBin", "avgByteDelta", "Avg Byte Delta by Comment Length",
        )
    with tabs[10]:
        result10 = queryEditSizeDistribution(df)
        queryCard("Q11 - Edit Size Distribution", "Breakdown of editSizeCategory across wikis.", len(result11) if not result11.empty else 0)
        if not result11.empty:
            st.dataframe(result11, use_container_width=True)
        else:
            st.info("No results. Collect more data and refresh.")
    with tabs[11]:
        renderQueryTab(
            "Q11 - Collaboration Clusters",
            "Pages with 3+ distinct editors — co-editing and network collaboration patterns.",
            queryCollaborationClusters(df), "title", "uniqueEditors", "Unique Editors per Page",
        )
    with tabs[12]:
        renderQueryTab(
            "Q12 - Outlier Edit Detection",
            "Edits with byte_delta above the 95th percentile — possible vandalism or major updates.",
            queryOutlierEdits(df), "title", "byte_delta", "Outlier Byte Delta by Page",
        )
        


if __name__ == "__main__":
    main()