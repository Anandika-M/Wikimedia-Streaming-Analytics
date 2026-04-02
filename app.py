# app.py
# Single consolidated Streamlit dashboard - light theme
# ML results (parquet/pkl) + Cassandra live queries (12 queries, dropdown)
#
# Run with:
#   streamlit run app.py

import os
import glob
import pickle
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from cassandra.cluster import Cluster
from cassandra.policies import RoundRobinPolicy

from queries import fetchAllEdits, QUERY_REGISTRY

warnings.filterwarnings("ignore")

# Paths
MODELS_DIR     = "/models"
PLOTS_DIR      = "/plots"
SILVER_PATH    = "/data/silver"
CASSANDRA_HOST = "127.0.0.1"
CASSANDRA_PORT = 9042
CASSANDRA_KS   = "wikipedia"

# Light theme plot style
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#f8f9fa",
    "axes.edgecolor":   "#dee2e6",
    "axes.labelcolor":  "#343a40",
    "xtick.color":      "#495057",
    "ytick.color":      "#495057",
    "text.color":       "#212529",
    "grid.color":       "#e9ecef",
    "grid.alpha":       0.8,
    "font.family":      "sans-serif",
})

BLUE   = "#3b82f6"
GREEN  = "#10b981"
RED    = "#ef4444"
ORANGE = "#f59e0b"
PURPLE = "#8b5cf6"


# Page config
st.set_page_config(
    page_title="Wikipedia Edit Analytics",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        background-color: #f8fafc;
        color: #1e293b;
    }
    .main { background-color: #f8fafc; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

    .page-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        border-left: 4px solid #3b82f6;
        padding-left: 0.75rem;
        margin: 1.5rem 0 0.8rem 0;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #3b82f6;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.25rem;
    }
    .query-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #8b5cf6;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .query-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #8b5cf6;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .query-desc {
        font-size: 0.88rem;
        color: #475569;
        margin-top: 0.2rem;
    }
    .status-ok   { background: #dcfce7; color: #16a34a; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
    .status-miss { background: #fee2e2; color: #dc2626; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }

    div[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { color: #64748b; font-weight: 500; padding: 0.6rem 1.2rem; }
    .stTabs [aria-selected="true"] { color: #3b82f6; border-bottom: 2px solid #3b82f6; }
    h1, h2, h3, h4 { color: #0f172a !important; }
    .stDataFrame { border: 1px solid #e2e8f0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# Loads a pkl file and returns its contents or None if missing
def loadPickle(fileName):
    filePath = os.path.join(MODELS_DIR, fileName)
    if not os.path.exists(filePath):
        return None
    with open(filePath, "rb") as f:
        return pickle.load(f)


# Loads a plot image from the plots directory or returns None
def loadPlot(fileName):
    filePath = os.path.join(PLOTS_DIR, fileName)
    if not os.path.exists(filePath):
        return None
    return Image.open(filePath)


# Loads silver parquet files into a pandas DataFrame with caching
@st.cache_data(ttl=120)
def loadSilverData():
    parquetFiles = glob.glob(f"{SILVER_PATH}/*.parquet")
    if not parquetFiles:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in parquetFiles], ignore_index=True)


# Loads all rows from Cassandra wiki_edits table with caching
@st.cache_data(ttl=60)
def loadCassandraData():
    try:
        cluster = Cluster(
            [CASSANDRA_HOST],
            port=CASSANDRA_PORT,
            load_balancing_policy=RoundRobinPolicy(),
            protocol_version=4,
        )
        session = cluster.connect(CASSANDRA_KS)
        df      = fetchAllEdits(session)
        cluster.shutdown()
        return df
    except Exception:
        return pd.DataFrame()


# Renders a KPI card with a large value and small label
def kpiCard(label, value):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)


# Displays a plot image or a soft info message if not found
def showPlot(fileName, caption=""):
    img = loadPlot(fileName)
    if img:
        st.image(img, caption=caption, use_container_width=True)
    else:
        st.info(f"Run step3_ml_sklearn.py to generate: {fileName}")


# Draws a horizontal bar chart in light theme colors
def barChart(df, xCol, yCol, title, color=BLUE, maxRows=15):
    if df.empty or xCol not in df.columns or yCol not in df.columns:
        st.caption("No chart data available.")
        return
    fig, ax = plt.subplots(figsize=(7, max(3, min(len(df), maxRows) * 0.4)))
    plotDf  = df.head(maxRows)
    ax.barh(plotDf[xCol].astype(str).str[:40], plotDf[yCol], color=color, edgecolor="none")
    ax.set_title(title, fontsize=11, fontweight="600", pad=8)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# Renders the sidebar with data source stats and model status
def renderSidebar(silverDf, cassandraDf):
    with st.sidebar:
        st.markdown("### Wikipedia\nEdit Analytics")
        st.markdown("---")

        st.markdown("**Data Sources**")
        silverRows = len(silverDf) if not silverDf.empty else 0
        cassRows   = len(cassandraDf) if not cassandraDf.empty else 0
        st.markdown(f"Parquet (Silver): **{silverRows:,}** rows")
        st.markdown(f"Cassandra: **{cassRows:,}** rows")

        st.markdown("---")
        st.markdown("**ML Models**")
        for fileName, name in [
            ("bot_classifier.pkl",   "Bot Classifier"),
            ("vandalism_kmeans.pkl", "KMeans Anomaly"),
            ("isolation_forest.pkl", "Isolation Forest"),
            ("trend_metrics.pkl",    "Trend Analysis"),
        ]:
            exists = os.path.exists(os.path.join(MODELS_DIR, fileName))
            badge  = '<span class="status-ok">READY</span>' if exists else '<span class="status-miss">MISSING</span>'
            st.markdown(f"{badge} {name}", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("Refresh All Data"):
            st.cache_data.clear()
            st.rerun()

        st.caption("Run step3_ml_sklearn.py for ML.\nRun step2_spark_streaming.py for live Cassandra data.")


# Renders the overview tab combining both data sources
def renderOverview(silverDf, cassandraDf, botMetrics, anomalyMetrics):
    st.markdown('<p class="page-title">Wikipedia Live Edit Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Wikimedia SSE Stream → Kafka → Spark Structured Streaming → Parquet + Cassandra → scikit-learn ML</p>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        kpiCard("Silver Rows", f"{len(silverDf):,}" if not silverDf.empty else "0")
    with col2:
        editRows = len(silverDf[silverDf["type"] == "edit"]) if not silverDf.empty and "type" in silverDf.columns else 0
        kpiCard("Article Edits", f"{editRows:,}")
    with col3:
        kpiCard("Cassandra Rows", f"{len(cassandraDf):,}" if not cassandraDf.empty else "0")
    with col4:
        auc = round(botMetrics["best_auc"], 4) if botMetrics else "N/A"
        kpiCard("Best AUC-ROC", auc)
    with col5:
        sil = round(anomalyMetrics["silhouette"], 4) if anomalyMetrics else "N/A"
        kpiCard("Silhouette", sil)
    with col6:
        anomCount = anomalyMetrics["anomalyCount"] if anomalyMetrics else "N/A"
        kpiCard("Anomalies", anomCount)

    st.markdown('<p class="section-title">Edit Distribution</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        showPlot("01_class_distribution.jpg", "Bot vs Human Distribution")
    with col2:
        showPlot("12_edit_size_distribution.jpg", "Edit Size Categories")


# Renders the bot classifier tab with all three model comparisons
def renderClassifier(botMetrics):
    st.markdown('<p class="page-title">Model A: Bot vs Human Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Logistic Regression, Decision Tree, and Random Forest compared. Best model selected by AUC-ROC.</p>', unsafe_allow_html=True)

    if not botMetrics:
        st.warning("No classifier metrics found. Run step3_ml_sklearn.py first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        lr = botMetrics.get("LogisticRegression", {}).get("auc", "N/A")
        kpiCard("Logistic Regression AUC", round(lr, 4) if isinstance(lr, float) else lr)
    with col2:
        dt = botMetrics.get("DecisionTree", {}).get("auc", "N/A")
        kpiCard("Decision Tree AUC", round(dt, 4) if isinstance(dt, float) else dt)
    with col3:
        rf = botMetrics.get("RandomForest", {}).get("auc", "N/A")
        kpiCard("Random Forest AUC", round(rf, 4) if isinstance(rf, float) else rf)

    st.markdown(f'<p class="section-title">Best Model: {botMetrics.get("best_name","N/A")} — AUC {round(botMetrics.get("best_auc",0),4)}</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        showPlot("05_classifier_comparison.jpg", "AUC Comparison")
        showPlot("03_roc_curve.jpg", "ROC Curve")
    with col2:
        showPlot("02_confusion_matrix.jpg", "Confusion Matrix")
        showPlot("04_feature_importances.jpg", "Feature Importances")

    st.markdown('<p class="section-title">Classification Report</p>', unsafe_allow_html=True)
    if "report" in botMetrics:
        st.code(botMetrics["report"], language="text")


# Renders the anomaly detection tab
def renderAnomaly(anomalyMetrics):
    st.markdown('<p class="page-title">Model B: Anomaly Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">KMeans clusters edits by behaviour. Isolation Forest flags unusual edits — large deletions, sudden spikes.</p>', unsafe_allow_html=True)

    if not anomalyMetrics:
        st.warning("No anomaly metrics found. Run step3_ml_sklearn.py first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        kpiCard("KMeans Clusters", 5)
    with col2:
        kpiCard("Silhouette Score", round(anomalyMetrics["silhouette"], 4))
    with col3:
        kpiCard("Anomalies Detected", anomalyMetrics["anomalyCount"])

    col1, col2 = st.columns(2)
    with col1:
        showPlot("06_kmeans_clusters.jpg", "KMeans Cluster Distribution")
        showPlot("08_anomaly_distribution.jpg", "Byte Delta — Normal vs Anomaly")
    with col2:
        showPlot("07_anomaly_scatter.jpg", "Anomaly Scatter — Isolation Forest")

    st.markdown('<p class="section-title">Cluster Profiles</p>', unsafe_allow_html=True)
    if "clusterProfile" in anomalyMetrics:
        st.dataframe(pd.DataFrame(anomalyMetrics["clusterProfile"]).round(2), use_container_width=True)


# Renders the trends tab with popular pages and edit patterns
def renderTrends(trendMetrics):
    st.markdown('<p class="page-title">Model C: Trend and Popular Page Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Most-edited pages, peak edit hours, active wikis, and bot ratio by language.</p>', unsafe_allow_html=True)

    if not trendMetrics:
        st.warning("No trend metrics found. Run step3_ml_sklearn.py first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        showPlot("09_top_pages.jpg", "Top 15 Most Edited Pages")
        showPlot("11_top_wikis.jpg", "Top 10 Wikis by Volume")
    with col2:
        showPlot("10_hourly_edits.jpg", "Edit Volume by Hour of Day")

    st.markdown('<p class="section-title">Top 10 Most Edited Pages</p>', unsafe_allow_html=True)
    if "topPages" in trendMetrics:
        pagesDf = pd.DataFrame(trendMetrics["topPages"])[["title", "editCount", "totalDelta", "botEdits"]].head(10)
        pagesDf.columns = ["Page Title", "Edit Count", "Total Byte Delta", "Bot Edits"]
        st.dataframe(pagesDf, use_container_width=True)

    st.markdown('<p class="section-title">Top Wikis Summary</p>', unsafe_allow_html=True)
    if "topWikis" in trendMetrics:
        st.dataframe(pd.DataFrame(trendMetrics["topWikis"]), use_container_width=True)


# Renders the Cassandra queries tab with a dropdown for all 12 queries
def renderCassandraQueries(cassandraDf):
    st.markdown('<p class="page-title">Cassandra Live Queries</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">12 analytical queries running against live Wikipedia edit data stored in Cassandra. Select a query from the dropdown.</p>', unsafe_allow_html=True)

    if cassandraDf.empty:
        st.warning("No data in Cassandra. Make sure Cassandra is running and step2_spark_streaming.py has written data.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        kpiCard("Cassandra Rows", f"{len(cassandraDf):,}")
    with col2:
        wikis = cassandraDf["wiki"].nunique() if "wiki" in cassandraDf.columns else 0
        kpiCard("Unique Wikis", wikis)
    with col3:
        users = cassandraDf["user_name"].nunique() if "user_name" in cassandraDf.columns else 0
        kpiCard("Unique Users", users)

    st.markdown('<p class="section-title">Select a Query</p>', unsafe_allow_html=True)

    # Dropdown built from QUERY_REGISTRY in queries.py
    queryLabels   = [q["label"] for q in QUERY_REGISTRY]
    selectedLabel = st.selectbox(
        "Choose an analytical query:",
        options=queryLabels,
        index=0,
        key="query_selector",
    )

    selectedQuery = next(q for q in QUERY_REGISTRY if q["label"] == selectedLabel)

    # Show the query description card
    st.markdown(f"""
    <div class="query-box">
        <div class="query-label">{selectedQuery["label"]}</div>
        <div class="query-desc">{selectedQuery["desc"]}</div>
    </div>
    """, unsafe_allow_html=True)

    # Run the selected query function
    with st.spinner("Running query..."):
        resultDf = selectedQuery["fn"](cassandraDf)

    if resultDf.empty:
        st.info("No results for this query. Collect more streaming data and click Refresh.")
        return

    st.markdown(f'<p class="section-title">Results — {len(resultDf)} rows</p>', unsafe_allow_html=True)

    xCol = selectedQuery.get("xCol")
    yCol = selectedQuery.get("yCol")

    # Q10 (Edit Size Distribution) has no chart - show full width table
    if selectedQuery["key"] == "q10":
        st.dataframe(resultDf, use_container_width=True)
    else:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.dataframe(resultDf, use_container_width=True)
        with col2:
            if xCol and yCol:
                barChart(resultDf, xCol, yCol, selectedQuery["label"].split(" - ")[1], color=PURPLE)
            else:
                st.caption("No chart available for this query.")


# Renders the raw silver data tab
def renderRawData(silverDf):
    st.markdown('<p class="page-title">Silver Layer Data</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Cleaned and enriched data from the Spark streaming pipeline stored as Parquet files.</p>', unsafe_allow_html=True)

    if silverDf.empty:
        st.warning("No silver data found.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        kpiCard("Total Rows", f"{len(silverDf):,}")
    with col2:
        kpiCard("Columns", len(silverDf.columns))
    with col3:
        wikis = silverDf["wiki"].nunique() if "wiki" in silverDf.columns else 0
        kpiCard("Unique Wikis", wikis)

    st.markdown('<p class="section-title">Schema</p>', unsafe_allow_html=True)
    schemaDf = pd.DataFrame({"Column": silverDf.columns, "Type": [str(t) for t in silverDf.dtypes]})
    st.dataframe(schemaDf, use_container_width=True)

    st.markdown('<p class="section-title">Sample — 50 rows</p>', unsafe_allow_html=True)
    displayCols   = ["wiki", "type", "title", "user", "bot", "byte_delta", "commentLen", "hourOfDay", "editSizeCategory", "isAnon"]
    availableCols = [c for c in displayCols if c in silverDf.columns]
    st.dataframe(silverDf[availableCols].head(50), use_container_width=True)


# Main function - loads all data and renders the full tabbed dashboard
def main():
    silverDf       = loadSilverData()
    cassandraDf    = loadCassandraData()
    botMetrics     = loadPickle("bot_metrics.pkl")
    anomalyMetrics = loadPickle("anomaly_metrics.pkl")
    trendMetrics   = loadPickle("trend_metrics.pkl")

    renderSidebar(silverDf, cassandraDf)

    tabs = st.tabs([
        "Overview",
        "Bot Classifier",
        "Anomaly Detection",
        "Trends",
        "Cassandra Queries",
        "Raw Data",
    ])

    with tabs[0]:
        renderOverview(silverDf, cassandraDf, botMetrics, anomalyMetrics)
    with tabs[1]:
        renderClassifier(botMetrics)
    with tabs[2]:
        renderAnomaly(anomalyMetrics)
    with tabs[3]:
        renderTrends(trendMetrics)
    with tabs[4]:
        renderCassandraQueries(cassandraDf)
    with tabs[5]:
        renderRawData(silverDf)


if __name__ == "__main__":
    main()