# step3_ml_sklearn.py
# Trains three ML models on silver parquet data using scikit-learn
# Saves models as .pkl files and plots as .jpg files
#
# Run with:
#   python3 step3_ml_sklearn.py
#
# Install dependencies:
#   pip install scikit-learn pandas numpy pyarrow matplotlib seaborn

import os
import glob
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix, roc_curve, silhouette_score,
)

warnings.filterwarnings("ignore")

# Paths
SILVER_PATH = "/data/silver"
MODELS_DIR  = "/models"
PLOTS_DIR   = "/plots"

RANDOM_SEED = 42
TEST_SPLIT  = 0.2


# Creates output directories for models and plots
def createOutputDirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("Models dir:", MODELS_DIR)
    print("Plots dir :", PLOTS_DIR)


# Loads all silver parquet files into a single pandas DataFrame
def loadSilverData():
    print("\nLoading silver parquet files from:", SILVER_PATH)
    parquetFiles = glob.glob(f"{SILVER_PATH}/*.parquet")
    if not parquetFiles:
        raise FileNotFoundError("No parquet files found in: " + SILVER_PATH)
    df = pd.concat([pd.read_parquet(f) for f in parquetFiles], ignore_index=True)
    print("Total rows loaded:", len(df))
    return df


# Filters to article edits and builds ML-ready feature columns
def prepareFeatures(df):
    editDf = df[(df["namespace"] == 0) & (df["type"] == "edit")].copy()
    editDf["byte_delta"] = editDf["byte_delta"].fillna(0)
    editDf["title_len"]  = editDf["title_len"].fillna(0)
    editDf["commentLen"] = editDf["commentLen"].fillna(0)
    editDf["latencyMs"]  = editDf["latencyMs"].fillna(0)
    editDf["hourOfDay"]  = editDf["hourOfDay"].fillna(0)
    editDf["dayOfWeek"]  = editDf["dayOfWeek"].fillna(1)
    editDf["isBot"]      = editDf["bot"].astype(int)
    editDf["isAnon"]     = editDf["isAnon"].astype(int)
    editDf["isMinor"]    = editDf["minor"].astype(int)
    editDf["absDelta"]   = editDf["byte_delta"].abs()
    editDf               = editDf.dropna(subset=["isBot"])
    print("Feature rows after filtering:", len(editDf))
    return editDf


# Saves a Python object to a pkl file in the models directory
def saveModel(obj, fileName):
    savePath = os.path.join(MODELS_DIR, fileName)
    with open(savePath, "wb") as f:
        pickle.dump(obj, f)
    print("Saved:", savePath)


# Saves a matplotlib figure as jpg and closes it to free memory
def savePlot(fig, fileName):
    savePath = os.path.join(PLOTS_DIR, fileName)
    fig.savefig(savePath, dpi=150, bbox_inches="tight", format="jpg")
    plt.close(fig)
    print("Saved plot:", savePath)


# Plots bot vs human class distribution bar chart
def plotClassDistribution(featureDf):
    counts  = featureDf["isBot"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars    = ax.bar(["Human", "Bot"], [counts.get(0, 0), counts.get(1, 0)], color=["#4C9BE8", "#E84C4C"], width=0.5)
    ax.set_title("Bot vs Human Edit Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Edits")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(int(bar.get_height())), ha="center", fontweight="bold")
    fig.tight_layout()
    savePlot(fig, "01_class_distribution.jpg")


# Plots confusion matrix heatmap for a classifier
def plotConfusionMatrix(yTest, yPred, modelName, fileName):
    cm      = confusion_matrix(yTest, yPred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Human", "Bot"], yticklabels=["Human", "Bot"])
    ax.set_title(f"Confusion Matrix - {modelName}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    fig.tight_layout()
    savePlot(fig, fileName)


# Plots ROC curve with AUC score for a binary classifier
def plotRocCurve(yTest, yProba, modelName, fileName):
    fpr, tpr, _ = roc_curve(yTest, yProba)
    auc         = roc_auc_score(yTest, yProba)
    fig, ax     = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C9BE8", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_title(f"ROC Curve - {modelName}", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    savePlot(fig, fileName)


# Plots horizontal bar chart of feature importances
def plotFeatureImportances(featureCols, importances, modelName, fileName):
    sortedIdx = np.argsort(importances)
    fig, ax   = plt.subplots(figsize=(7, 5))
    ax.barh([featureCols[i] for i in sortedIdx], [importances[i] for i in sortedIdx], color="#4C9BE8")
    ax.set_title(f"Feature Importances - {modelName}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    fig.tight_layout()
    savePlot(fig, fileName)


# Trains Logistic Regression, Decision Tree, and Random Forest for bot detection
# Picks the best model by AUC and saves it as pkl
def trainBotClassifier(featureDf):
    print("\n--- Model A: Bot vs Human Classifier ---")

    featureCols = ["byte_delta", "absDelta", "title_len", "commentLen", "isAnon", "isMinor", "hourOfDay", "dayOfWeek"]
    X = featureDf[featureCols].values
    y = featureDf["isBot"].values

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
    print("Train:", len(xTrain), "| Test:", len(xTest))

    # Train all three classifiers
    models = {
        "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=500, random_state=RANDOM_SEED))]),
        "DecisionTree":       Pipeline([("scaler", StandardScaler()), ("model", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED))]),
        "RandomForest":       Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=50, max_depth=6, random_state=RANDOM_SEED))]),
    }

    results = {}
    for name, pipeline in models.items():
        pipeline.fit(xTrain, yTrain)
        proba         = pipeline.predict_proba(xTest)[:, 1]
        auc           = roc_auc_score(yTest, proba)
        results[name] = {"pipeline": pipeline, "proba": proba, "auc": auc}
        print(f"  {name:<22s} AUC: {auc:.4f}")

    # Pick best by AUC
    bestName  = max(results, key=lambda k: results[k]["auc"])
    bestModel = results[bestName]["pipeline"]
    bestProba = results[bestName]["proba"]
    bestAuc   = results[bestName]["auc"]
    bestPred  = bestModel.predict(xTest)

    print(f"\nBest: {bestName} AUC={round(bestAuc, 4)}")
    print(classification_report(yTest, bestPred, target_names=["Human", "Bot"]))

    # Generate all plots
    plotClassDistribution(featureDf)
    plotConfusionMatrix(yTest, bestPred, bestName, "02_confusion_matrix.jpg")
    plotRocCurve(yTest, bestProba, bestName, "03_roc_curve.jpg")
    if hasattr(bestModel.named_steps["model"], "feature_importances_"):
        plotFeatureImportances(featureCols, bestModel.named_steps["model"].feature_importances_, bestName, "04_feature_importances.jpg")

    # AUC comparison bar chart across all three models
    fig, ax = plt.subplots(figsize=(6, 4))
    names   = list(results.keys())
    aucs    = [results[n]["auc"] for n in names]
    bars    = ax.bar(["Logistic\nRegression", "Decision\nTree", "Random\nForest"], aucs, color=["#E8A84C", "#4CE8A8", "#4C9BE8"])
    ax.set_title("Classifier AUC Comparison", fontsize=13, fontweight="bold")
    ax.set_ylabel("AUC-ROC Score")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.4f}", ha="center", fontweight="bold")
    fig.tight_layout()
    savePlot(fig, "05_classifier_comparison.jpg")

    metrics = {n: {"auc": results[n]["auc"]} for n in results}
    metrics["best_name"] = bestName
    metrics["best_auc"]  = bestAuc
    metrics["report"]    = classification_report(yTest, bestPred, target_names=["Human", "Bot"])

    saveModel(bestModel, "bot_classifier.pkl")
    saveModel(metrics,   "bot_metrics.pkl")
    return bestModel, metrics


# Trains KMeans clustering and Isolation Forest for anomaly detection
def trainAnomalyDetector(featureDf):
    print("\n--- Model B: Anomaly Detection ---")

    featureCols = ["byte_delta", "absDelta", "title_len", "commentLen", "isAnon", "isMinor"]
    X = featureDf[featureCols].values

    # KMeans pipeline
    kmeansPipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", KMeans(n_clusters=5, random_state=RANDOM_SEED, n_init=10, max_iter=20)),
    ])
    kmeansPipeline.fit(X)
    clusterLabels = kmeansPipeline.predict(X)
    xScaled       = kmeansPipeline.named_steps["scaler"].transform(X)
    silhouette    = silhouette_score(xScaled, clusterLabels, sample_size=2000, random_state=RANDOM_SEED)
    print("KMeans silhouette score:", round(silhouette, 4))

    # Isolation Forest
    iForest       = IsolationForest(n_estimators=100, contamination=0.05, random_state=RANDOM_SEED)
    iForest.fit(xScaled)
    anomalyLabels = iForest.predict(xScaled)
    anomalyCount  = int((anomalyLabels == -1).sum())
    print("Isolation Forest anomalies detected:", anomalyCount)

    profileDf            = featureDf.copy()
    profileDf["cluster"] = clusterLabels
    profileDf["anomaly"] = (anomalyLabels == -1).astype(int)

    clusterProfile = profileDf.groupby("cluster").agg(
        count=("byte_delta", "count"),
        avgDelta=("byte_delta", "mean"),
        avgComment=("commentLen", "mean"),
        anonRatio=("isAnon", "mean"),
    ).round(2)
    print("\nCluster profiles:\n", clusterProfile)

    # Cluster bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    counts  = profileDf["cluster"].value_counts().sort_index()
    ax.bar([f"Cluster {i}" for i in counts.index], counts.values, color="#4C9BE8")
    ax.set_title("Edit Clusters - KMeans (k=5)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Edits")
    fig.tight_layout()
    savePlot(fig, "06_kmeans_clusters.jpg")

    # Anomaly scatter plot
    normal = profileDf[profileDf["anomaly"] == 0]
    anoms  = profileDf[profileDf["anomaly"] == 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(normal["byte_delta"], normal["commentLen"], alpha=0.3, s=10, color="#4C9BE8", label="Normal")
    ax.scatter(anoms["byte_delta"],  anoms["commentLen"],  alpha=0.7, s=20, color="#E84C4C", label=f"Anomaly ({anomalyCount})")
    ax.set_title("Anomaly Detection - Isolation Forest", fontsize=13, fontweight="bold")
    ax.set_xlabel("Byte Delta (edit size)")
    ax.set_ylabel("Comment Length")
    ax.legend()
    fig.tight_layout()
    savePlot(fig, "07_anomaly_scatter.jpg")

    # Byte delta distribution normal vs anomaly
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(normal["byte_delta"].clip(-3000, 3000), bins=50, alpha=0.6, color="#4C9BE8", label="Normal", density=True)
    ax.hist(anoms["byte_delta"].clip(-3000, 3000),  bins=20, alpha=0.8, color="#E84C4C", label="Anomaly", density=True)
    ax.set_title("Byte Delta Distribution - Normal vs Anomaly", fontsize=13, fontweight="bold")
    ax.set_xlabel("Byte Delta")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    savePlot(fig, "08_anomaly_distribution.jpg")

    metrics = {"silhouette": silhouette, "anomalyCount": anomalyCount, "clusterProfile": clusterProfile.to_dict()}
    saveModel(kmeansPipeline, "vandalism_kmeans.pkl")
    saveModel(iForest,        "isolation_forest.pkl")
    saveModel(metrics,        "anomaly_metrics.pkl")
    return kmeansPipeline, iForest, metrics


# Analyzes edit trends - top pages, hourly patterns, wiki distribution
def analyzeTrends(df):
    print("\n--- Model C: Trend and Popular Page Analysis ---")

    editDf = df[(df["namespace"] == 0) & (df["type"] == "edit")].copy()
    editDf["byte_delta"] = editDf["byte_delta"].fillna(0)
    editDf["isBot"]      = editDf["bot"].astype(int)
    editDf["isAnon"]     = editDf["isAnon"].astype(int)

    # Top 15 most edited pages
    topPages = (
        editDf.groupby("title")
        .agg(editCount=("title", "count"), totalDelta=("byte_delta", "sum"), botEdits=("bot", "sum"))
        .reset_index()
        .sort_values("editCount", ascending=False)
        .head(15)
    )
    print("Top 5 pages:", topPages["title"].head(5).tolist())

    # Hourly edit volume
    hourlyEdits = editDf.groupby("hourOfDay").size().reset_index(name="editCount")

    # Top wikis by edit count
    topWikis = (
        editDf.groupby("wikiLang")
        .agg(editCount=("title", "count"), botEdits=("bot", "sum"))
        .reset_index()
        .sort_values("editCount", ascending=False)
        .head(10)
    )

    # Edit size category counts
    sizeDist = editDf["editSizeCategory"].value_counts()

    # Bot ratio per wiki
    botRatio = (
        editDf.groupby("wikiLang")
        .agg(totalEdits=("title", "count"), botEdits=("bot", "sum"))
        .assign(botRatio=lambda x: x["botEdits"] / x["totalEdits"])
        .reset_index()
        .sort_values("botRatio", ascending=False)
        .head(10)
    )

    # Plot top 15 most edited pages
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(topPages["title"].str[:45], topPages["editCount"], color="#4C9BE8")
    ax.set_title("Top 15 Most Edited Pages", fontsize=13, fontweight="bold")
    ax.set_xlabel("Edit Count")
    ax.invert_yaxis()
    fig.tight_layout()
    savePlot(fig, "09_top_pages.jpg")

    # Plot hourly edit pattern
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hourlyEdits["hourOfDay"], hourlyEdits["editCount"], marker="o", color="#4C9BE8", linewidth=2)
    ax.fill_between(hourlyEdits["hourOfDay"], hourlyEdits["editCount"], alpha=0.2, color="#4C9BE8")
    ax.set_title("Edit Volume by Hour of Day (UTC)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Edit Count")
    ax.set_xticks(range(0, 24))
    fig.tight_layout()
    savePlot(fig, "10_hourly_edits.jpg")

    # Plot top wikis
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(topWikis["wikiLang"], topWikis["editCount"], color="#4C9BE8")
    ax.set_title("Top 10 Wikis by Edit Volume", fontsize=13, fontweight="bold")
    ax.set_xlabel("Wiki Language")
    ax.set_ylabel("Edit Count")
    fig.tight_layout()
    savePlot(fig, "11_top_wikis.jpg")

    # Plot edit size distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ["#4C9BE8", "#E84C4C", "#4CE8A8", "#E8A84C", "#A84CE8"]
    ax.bar(sizeDist.index, sizeDist.values, color=colors[:len(sizeDist)])
    ax.set_title("Edit Size Category Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    fig.tight_layout()
    savePlot(fig, "12_edit_size_distribution.jpg")

    # Plot bot ratio per wiki
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(botRatio["wikiLang"], botRatio["botRatio"], color="#E84C4C")
    ax.set_title("Bot Edit Ratio by Wiki Language", fontsize=13, fontweight="bold")
    ax.set_xlabel("Wiki Language")
    ax.set_ylabel("Bot Ratio (0-1)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    savePlot(fig, "13_bot_ratio_by_wiki.jpg")

    trendMetrics = {
        "topPages":      topPages.to_dict(orient="records"),
        "hourlyEdits":   hourlyEdits.to_dict(orient="records"),
        "topWikis":      topWikis.to_dict(orient="records"),
        "sizeDist":      sizeDist.to_dict(),
        "botRatio":      botRatio.to_dict(orient="records"),
    }
    saveModel(trendMetrics, "trend_metrics.pkl")
    return trendMetrics


# Main function - runs all analyses and prints final summary
def main():
    createOutputDirs()

    df        = loadSilverData()
    featureDf = prepareFeatures(df)

    modelA, metricsA            = trainBotClassifier(featureDf)
    kmeansB, iforestB, metricsB = trainAnomalyDetector(featureDf)
    metricsC                    = analyzeTrends(df)

    print("\n=== TRAINING SUMMARY ===")
    print("Model A - Best Classifier  :", metricsA["best_name"], "| AUC:", round(metricsA["best_auc"], 4))
    print("Model B - KMeans Silhouette:", round(metricsB["silhouette"], 4))
    print("Model B - Anomalies Found  :", metricsB["anomalyCount"])
    print("Model C - Top Page         :", metricsC["topPages"][0]["title"] if metricsC["topPages"] else "N/A")
    print("Models saved to:", MODELS_DIR)
    print("Plots saved to :", PLOTS_DIR)


if __name__ == "__main__":
    main()