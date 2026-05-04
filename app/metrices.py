import pandas as pd
import numpy as np
import ast
from collections import Counter

# Load CSV
df = pd.read_csv("/mnt/data/evaluation_details_k10_20260216_145510.csv")

# Parse lists safely
df["ground_truth_movies"] = df["ground_truth_movies"].apply(ast.literal_eval)
df["recommended_movies"] = df["recommended_movies"].apply(ast.literal_eval)

K = 10

# ---------- Helper Functions ----------

def precision_at_k(rec, gt, k=10):
    return len(set(rec[:k]).intersection(set(gt))) / k

def dcg_at_k(rec, gt, k=10):
    score = 0
    for i, item in enumerate(rec[:k]):
        if item in gt:
            score += 1 / np.log2(i + 2)
    return score

def ndcg_at_k(rec, gt, k=10):
    dcg = dcg_at_k(rec, gt, k)
    ideal_hits = min(len(gt), k)
    idcg = sum([1 / np.log2(i + 2) for i in range(ideal_hits)])
    return dcg / idcg if idcg > 0 else 0

# ---------- OUR MODEL ----------

df["precision@10"] = df.apply(lambda x: precision_at_k(x["recommended_movies"], x["ground_truth_movies"], K), axis=1)
df["ndcg@10"] = df.apply(lambda x: ndcg_at_k(x["recommended_movies"], x["ground_truth_movies"], K), axis=1)

ours_metrics = {
    "Hit@10": df["hit"].mean(),
    "Recall@10": df["recall"].mean(),
    "Precision@10": df["precision@10"].mean(),
    "NDCG@10": df["ndcg@10"].mean()
}

# ---------- RANDOM BASELINE ----------

def random_metrics(row, iterations=30):
    rec_original = row["recommended_movies"]
    gt = row["ground_truth_movies"]
    
    hits, recalls, precisions, ndcgs = [], [], [], []
    
    for _ in range(iterations):
        rec = rec_original.copy()
        np.random.shuffle(rec)
        hit_set = set(rec[:K]).intersection(set(gt))
        
        hits.append(1 if len(hit_set) > 0 else 0)
        recalls.append(len(hit_set) / len(gt) if len(gt) > 0 else 0)
        precisions.append(len(hit_set) / K)
        ndcgs.append(ndcg_at_k(rec, gt, K))
        
    return np.mean(hits), np.mean(recalls), np.mean(precisions), np.mean(ndcgs)

rand_values = df.apply(lambda row: random_metrics(row), axis=1)
df[["rand_hit","rand_recall","rand_precision","rand_ndcg"]] = pd.DataFrame(rand_values.tolist(), index=df.index)

random_metrics_summary = {
    "Hit@10": df["rand_hit"].mean(),
    "Recall@10": df["rand_recall"].mean(),
    "Precision@10": df["rand_precision"].mean(),
    "NDCG@10": df["rand_ndcg"].mean()
}

# ---------- POPULARITY BASELINE ----------

all_recs = [movie for sublist in df["recommended_movies"] for movie in sublist]
popular_movies = [movie for movie, _ in Counter(all_recs).most_common(K)]

def popularity_metrics(row):
    gt = row["ground_truth_movies"]
    hit_set = set(popular_movies).intersection(set(gt))
    
    hit = 1 if len(hit_set) > 0 else 0
    recall = len(hit_set) / len(gt) if len(gt) > 0 else 0
    precision = len(hit_set) / K
    ndcg = ndcg_at_k(popular_movies, gt, K)
    
    return hit, recall, precision, ndcg

pop_values = df.apply(lambda row: popularity_metrics(row), axis=1)
df[["pop_hit","pop_recall","pop_precision","pop_ndcg"]] = pd.DataFrame(pop_values.tolist(), index=df.index)

pop_metrics_summary = {
    "Hit@10": df["pop_hit"].mean(),
    "Recall@10": df["pop_recall"].mean(),
    "Precision@10": df["pop_precision"].mean(),
    "NDCG@10": df["pop_ndcg"].mean()
}

# ---------- SUMMARY TABLE ----------

summary_df = pd.DataFrame([
    {"Model": "Random", **random_metrics_summary},
    {"Model": "Popularity", **pop_metrics_summary},
    {"Model": "Graph-CF (Ours)", **ours_metrics}
])

summary_df
