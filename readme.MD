#  Wikimedia Streaming Analytics

A real-time big data pipeline that ingests live Wikipedia edits via the **Wikimedia SSE stream**, processes them through **Kafka -> Spark Structured Streaming**, stores results in **Cassandra + Parquet**, and surfaces insights through a **dual Streamlit dashboard** backed by **scikit-learn ML models**.

---

##  Architecture

```
Wikimedia SSE Stream
        |
        v
      Kafka
        |
        v
Spark Structured Streaming
        |
    ┌───┴────────┐
    v            v
 Parquet      Cassandra
(Silver Layer) (wikipedia KS)
    │            │
    └────┬───────┘
         v
  scikit-learn ML
  (Bot Classifier, KMeans, Isolation Forest)
         │
         v
  Streamlit Dashboards
  (app.py , app_cassandra.py)
```

---

## Dashboards

### `app.py` - Unified Analytics Dashboard
A full-featured, light-themed Streamlit app combining ML results and live Cassandra queries into a single tabbed interface.

| Tab | Description |
|---|---|
| **Overview** | KPI cards: silver rows, edits, Cassandra rows, AUC-ROC, silhouette score, anomaly count |
| **Bot Classifier** | Logistic Regression vs Decision Tree vs Random Forest - AUC comparison, ROC curve, confusion matrix, feature importances |
| **Anomaly Detection** | KMeans cluster profiles + Isolation Forest scatter - flags large deletions and edit spikes |
| **Trends** | Top 15 most-edited pages, edit volume by hour, top wikis by volume |
| **Cassandra Queries** | Dropdown of 12 analytical queries with live results and bar charts |
| **Raw Data** | Schema explorer and 50-row sample of the silver Parquet layer |

### `app_cassandra.py` - Deep-Dive Cassandra Dashboard
A dashboard with 12 dedicated query tabs, each with its own results table and chart.

---

##  ML Models

Three models trained in `step3_ml_sklearn.py`:

### Model A - Bot vs Human Classifier
Classifies each edit as bot or human using:
- Logistic Regression
- Decision Tree
- Random Forest *(selected by best AUC-ROC)*

**Artifacts:** `bot_classifier.pkl`, `bot_metrics.pkl`

### Model B - Anomaly Detection
- **KMeans (5 clusters)** - groups edits by behavioral profile
- **Isolation Forest** - flags outlier edits (large deletions, sudden spikes)

**Artifacts:** `vandalism_kmeans.pkl`, `isolation_forest.pkl`, `anomaly_metrics.pkl`

### Model C - Trend Analysis
- Most-edited pages, peak edit hours, active wikis, bot ratio by language

**Artifacts:** `trend_metrics.pkl`

---

##  Cassandra Queries (12 total)

| # | Query | Description |
|---|---|---|
| Q1 | Edit Spike Detection | Pages edited 2× above average - trending topics |
| Q2 | Median Edit Size per Wiki | Robust byte delta distribution per wiki |
| Q3 | Multi-User Rapid Edit Pages | Pages with 3+ users in a 1-hour window |
| Q4 | Repeated Edits by Same User | Same-user edits on a page 3+ times - edit wars |
| Q5 | Cross-Language Contributors | Users active across 2+ wiki languages |
| Q6 | Edit Session Identification | Sessions grouped by 30-min idle gap |
| Q7 | Longest Editing Streak | Max edits by a user in a single day |
| Q8 | High-Impact Contributors | Non-bot users ranked by avg absolute byte change |
| Q9 | Comment Length vs Edit Size | Correlation between comment length and edit size |
| Q10 | Edit Size Distribution | Breakdown of `editSizeCategory` across wikis |
| Q11 | Collaboration Clusters | Pages with 3+ distinct editors |
| Q12 | Outlier Edit Detection | Edits above the 95th percentile byte delta |


---

##  Project Structure

```
spark_wikipedia/ 
├── app.py                         # Unified dashboard (light theme)
├── app_cassandra.py               # Deep-dive Cassandra dashboard 
├──docker-compose.yml              # Compose file for kafka and cassandra
├── queries.py                     # QUERY_REGISTRY + all 12 query functions
├── setup_cssandra.py              # Creates the keyspace and table in Cassandra
├── step2_spark_streaming.py       # Spark job: Kafka -> Parquet + Cassandra
├── step3_ml_sklearn.py            # scikit-learn training: classifier + anomaly + trends
├── wikimedia_kafka_producer.py    # start kafka and fetch streaming data
├── models/
│   ├── bot_classifier.pkl
│   ├── bot_metrics.pkl
│   ├── vandalism_kmeans.pkl
│   ├── isolation_forest.pkl
│   ├── anomaly_metrics.pkl
│   └── trend_metrics.pkl
├── plots/                        # Auto-generated plot images (01–12)
└── data/
    └── silver/                   # Parquet files from Spark streaming
```

---

## Setup & Running

### Prerequisites
- Python 3.9+
- Apache Kafka
- Apache Spark
- Apache Cassandra (running locally on `127.0.0.1:9042`)
- Keyspace: `wikipedia`

### Install Dependencies
```bash
pip install streamlit pandas numpy matplotlib pillow cassandra-driver scikit-learn pyarrow
```

### Step 1 - Start the Streaming Pipeline
```bash
python step2_spark_streaming.py
```
Reads from Kafka, enriches data (byte delta, hour of day, edit size category, anon flag), and writes to both Parquet (silver layer) and Cassandra.

### Step 2 - Train ML Models
```bash
python step3_ml_sklearn.py
```
Trains all three models, saves `.pkl` files to `models/`, and generates 12 plot images to `plots/`.

### Step 3 - Launch Dashboards
```bash
# Unified dashboard
streamlit run app.py

# Cassandra deep-dive dashboard
streamlit run app_cassandra.py
```

The unified dashboard looke like :

![Project Preview](/assets/preview.png)
---

##  Configuration

All paths and connection settings are set at the top of each file:

```python
MODELS_DIR     = "/path/to/models"
PLOTS_DIR      = "/path/to/plots"
SILVER_PATH    = "/path/to/data/silver"
CASSANDRA_HOST = "127.0.0.1"
CASSANDRA_PORT = 9042
CASSANDRA_KS   = "wikipedia"
```

---

##  Key Metrics

| Metric | Source |
|---|---|
| AUC-ROC (best classifier) | `bot_metrics.pkl` -> `best_auc` |
| Silhouette Score | `anomaly_metrics.pkl` -> `silhouette` |
| Anomaly Count | `anomaly_metrics.pkl` -> `anomalyCount` |
| Cluster Profiles | `anomaly_metrics.pkl` -> `clusterProfile` |
| Top Edited Pages | `trend_metrics.pkl` -> `topPages` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Ingestion | Wikimedia SSE -> Kafka |
| Processing | Apache Spark Structured Streaming |
| Storage | Apache Cassandra, Apache Parquet |
| ML | scikit-learn (LR, DT, RF, KMeans, Isolation Forest) |
| Visualization | Streamlit, Matplotlib |
| Language | Python 3.9+ |
