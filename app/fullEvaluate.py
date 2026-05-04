import random
from datetime import datetime
from math import log2

import pandas as pd
from tqdm import tqdm

from app.graph import graph  # همان import پروژه شما


class MovieRecommenderEvaluator:
    """
    Extended version of MovieRecommenderEvaluator:
    - Reads everything directly from Neo4j via graph.query (same style as your evaluate.py)
    - Train/Test split inside graph with TEST_RATED
    - Offline evaluation for multiple models:
        * Graph-CF (ours)
        * Random baseline
        * Popularity baseline
    - Metrics:
        * Hit@K
        * Recall@K
        * Precision@K
        * NDCG@K  (binary relevance)
    - Saves detailed CSV (per-user) like your original code
    """

    def __init__(self, graph):
        self.graph = graph
        self.driver = graph._driver
        self.k = 10  # default Top-K
        self._cached_popular = {}  # cache per k -> list[movieId]

    # ---------------------------------------------------------------------
    # 0) Data sanity + constraints
    # ---------------------------------------------------------------------
    def check_data(self):
        """
        Checks if data exists in the Neo4j graph instead of loading from CSV.
        """
        check_query = "MATCH ()-[r:RATED]->() RETURN count(r) > 0 as exists"
        result = self.graph.query(check_query)

        if result and result[0]["exists"]:
            print("Data already exists in the graph. Skipping CSV load.")
        else:
            print("WARNING: No RATED relationships found in the graph!")
            print("Please make sure you have loaded your movie data into Neo4j first.")
            # raise Exception("No data found in Neo4j")

        # Constraints (IF NOT EXISTS)
        try:
            self.graph.query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE"
            )
            self.graph.query(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE"
            )
            print("Constraints verified.")
        except Exception as e:
            print(f"Note: Could not create constraints (they might exist): {e}")

    # ---------------------------------------------------------------------
    # 1) Train/Test split inside Neo4j (same logic as your evaluate.py)
    # ---------------------------------------------------------------------
    def cleanup(self):
        """
        Restores graph to original state by converting TEST_RATED back to RATED.
        """
        print("Cleaning up test data and restoring original graph...")
        self.graph.query(
            """
            MATCH (u:User)-[tr:TEST_RATED]->(m:Movie)
            MERGE (u)-[r:RATED]->(m)
            SET r.rating = tr.rating, r.timestamp = tr.timestamp
            DELETE tr
            """
        )
        print("Graph restored.")

    def prepare_test_set(self, test_ratio=0.2):
        """
        Splits the data into Train and Test sets based on existing data in Neo4j.
        Creates TEST_RATED edges by masking a portion of RATED edges.
        """
        print(f"Preparing test set (splitting {test_ratio * 100}% of data)...")

        # Restore state first (if prior run left TEST_RATED edges)
        self.cleanup()

        # Assign random number to relationships
        self.graph.query(
            """
            MATCH (u:User)-[r:RATED]->(m:Movie)
            SET r.random = rand()
            """
        )

        # Split logic (exact same structure as your original code)
        split_query = f"""
            MATCH (u:User)-[r:RATED]->(m:Movie)
            WITH u, m, r
            ORDER BY r.random
            WITH u, collect(r) as ratings
            WITH u, ratings, size(ratings) AS rating_count,
                 toInteger(size(ratings) * {test_ratio}) as raw_test_count
            WITH u, ratings,
                 CASE
                     WHEN rating_count <= 1 THEN 0
                     WHEN raw_test_count = 0 THEN 1
                     ELSE raw_test_count
                 END AS test_count
            UNWIND ratings[0..test_count] as test_rel
            MATCH (u)-[test_rel]->(m)
            CREATE (u)-[tr:TEST_RATED]->(m)
            SET tr.rating = test_rel.rating, tr.timestamp = test_rel.timestamp
            DELETE test_rel
        """
        self.graph.query(split_query)

        stats = self.graph.query(
            """
            MATCH ()-[tr:TEST_RATED]->()
            RETURN count(tr) AS test_count
            """
        )
        print(f"Created TEST_RATED relationships: {stats[0]['test_count'] if stats else 0}")

        # remove random attribute
        self.graph.query("MATCH ()-[r:RATED]->() REMOVE r.random")
        print("Test set prepared. Original ratings masked.")

    # ---------------------------------------------------------------------
    # 2) Core recommender (Graph-CF) — same as your evaluate.py
    # ---------------------------------------------------------------------
    def get_recommendations_cf(self, user_id):
        """
        Collaborative Filtering recommendations (ours).
        """
        query = """
        MATCH (target:User)
        WHERE elementId(target) = $userEid
        MATCH (target)-[:RATED]->(m:Movie)<-[:RATED]-(other:User)
        WITH target, other, count(m) as intersection
        WHERE intersection > 0
        MATCH (other)-[r:RATED]->(rec:Movie)
        WHERE NOT (target)-[:RATED]->(rec)
          AND NOT (target)-[:TEST_RATED]->(rec)
        WITH rec, count(r) as freq
        RETURN coalesce(toString(rec.id), rec.title, elementId(rec)) AS movieId
        ORDER BY freq DESC
        LIMIT $k
        """
        results = self.graph.query(query, {"userEid": user_id, "k": self.k})
        return [r["movieId"] for r in results]

    # ---------------------------------------------------------------------
    # 3) Baselines (Random / Popularity) — graph-driven
    # ---------------------------------------------------------------------
    def _get_unseen_movie_ids(self, user_id):
        """
        Returns candidate movies that user has NOT rated in train (RATED) and NOT in test (TEST_RATED).
        Fully read from Neo4j.
        """
        candidate_query = """
        MATCH (m:Movie)
        WHERE NOT EXISTS {
            MATCH (u:User)
            WHERE elementId(u) = $userEid
            MATCH (u)-[:RATED|:TEST_RATED]->(m)
        }
        RETURN coalesce(toString(m.id), m.title, elementId(m)) AS movieId
        """
        rows = self.graph.query(candidate_query, {"userEid": user_id})
        return [r["movieId"] for r in rows]

    def get_recommendations_random(self, user_id):
        """
        Random Baseline: uniformly sample K movies from unseen pool.
        """
        candidates = self._get_unseen_movie_ids(user_id)
        if not candidates:
            return []
        if len(candidates) <= self.k:
            return candidates
        return random.sample(candidates, self.k)

    def _get_popular_movies_global(self, k):
        """
        Popularity list computed globally (by count of incoming :RATED edges).
        Cached by k.
        """
        if k in self._cached_popular:
            return self._cached_popular[k]

        pop_query = """
        MATCH (m:Movie)<-[r:RATED]-(:User)
        WITH m, count(r) AS freq
        RETURN coalesce(toString(m.id), m.title, elementId(m)) AS movieId
        ORDER BY freq DESC
        LIMIT $k
        """
        rows = self.graph.query(pop_query, {"k": k})
        popular = [r["movieId"] for r in rows]
        self._cached_popular[k] = popular
        return popular

    def get_recommendations_popularity(self, user_id):
        """
        Popularity Baseline: top-K most-rated movies globally,
        excluding those already rated by the user (train/test).
        """
        # get a bit more than k to account for filtering
        # (safe fallback: if filtering removes many, we'll refill)
        initial_k = max(self.k * 5, self.k)

        pop_query = """
        MATCH (m:Movie)<-[r:RATED]-(:User)
        WITH m, count(r) AS freq
        ORDER BY freq DESC
        WITH collect(m) AS ms
        MATCH (u:User)
        WHERE elementId(u) = $userEid
        WITH u, ms
        UNWIND ms AS m
        WITH u, m
        WHERE NOT (u)-[:RATED|:TEST_RATED]->(m)
        RETURN coalesce(toString(m.id), m.title, elementId(m)) AS movieId
        LIMIT $k
        """
        rows = self.graph.query(pop_query, {"userEid": user_id, "k": self.k})
        recs = [r["movieId"] for r in rows]

        # If somehow we got fewer than k (rare), try to refill from global popularity cache with filtering
        if len(recs) < self.k:
            popular_global = self._get_popular_movies_global(initial_k)
            # need user seen set
            seen_query = """
            MATCH (u:User)
            WHERE elementId(u) = $userEid
            MATCH (u)-[:RATED|:TEST_RATED]->(m:Movie)
            RETURN coalesce(toString(m.id), m.title, elementId(m)) AS movieId
            """
            seen = set([r["movieId"] for r in self.graph.query(seen_query, {"userEid": user_id})])
            for mid in popular_global:
                if mid not in seen and mid not in recs:
                    recs.append(mid)
                if len(recs) >= self.k:
                    break

        return recs[: self.k]

    # ---------------------------------------------------------------------
    # 4) Metric helpers (binary relevance)
    # ---------------------------------------------------------------------
    @staticmethod
    def _precision_at_k(recs, gt, k):
        if k <= 0:
            return 0.0
        return len(set(recs[:k]).intersection(gt)) / float(k)

    @staticmethod
    def _recall_at_k(recs, gt, k):
        if not gt:
            return 0.0
        return len(set(recs[:k]).intersection(gt)) / float(len(gt))

    @staticmethod
    def _hit_at_k(recs, gt, k):
        return 1 if len(set(recs[:k]).intersection(gt)) > 0 else 0

    @staticmethod
    def _dcg_at_k(recs, gt, k):
        """
        DCG with binary relevance: rel_i = 1 if rec at rank i in GT else 0
        """
        score = 0.0
        for i, item in enumerate(recs[:k]):
            if item in gt:
                score += 1.0 / log2(i + 2)  # i starts at 0 => rank 1 uses log2(2)
        return score

    @classmethod
    def _ndcg_at_k(cls, recs, gt, k):
        dcg = cls._dcg_at_k(recs, gt, k)
        ideal_hits = min(len(gt), k)
        if ideal_hits <= 0:
            return 0.0
        idcg = sum([1.0 / log2(i + 2) for i in range(ideal_hits)])
        return dcg / idcg if idcg > 0 else 0.0

    # ---------------------------------------------------------------------
    # 5) Unified recommendation interface
    # ---------------------------------------------------------------------
    def get_recommendations_by_model(self, user_id, model_name):
        """
        model_name in: ours / graph-cf / cf, random, popularity
        """
        name = (model_name or "").strip().lower()

        if name in ["ours", "graph-cf", "cf", "collaborative", "collaborative-filtering"]:
            return self.get_recommendations_cf(user_id)
        if name in ["random", "random-baseline", "baseline-random"]:
            return self.get_recommendations_random(user_id)
        if name in ["popularity", "popular", "popularity-baseline", "baseline-popularity"]:
            return self.get_recommendations_popularity(user_id)

        raise ValueError(f"Unknown model_name: {model_name}")

    # ---------------------------------------------------------------------
    # 6) Evaluation loop (graph-driven) + save CSV
    # ---------------------------------------------------------------------
    def run_evaluation(self, k=10, model_name="ours", save_csv=True):
        """
        Runs offline evaluation for a specific model_name at top-k.
        Returns aggregate metrics dict + detailed dataframe (if any).
        """
        self.k = k
        print(f"\nStarting Evaluation ({model_name}) @ {k}...")

        # Users who have test data
        users_with_test = self.graph.query(
            """
            MATCH (u:User)-[:TEST_RATED]->()
            RETURN DISTINCT elementId(u) AS userId,
                   coalesce(toString(u.id), u.name, elementId(u)) AS userLabel
            """
        )

        if not users_with_test:
            print("No test data found. Did you run prepare_test_set?")
            return None, None

        user_ids = [u["userId"] for u in users_with_test]
        user_labels = {u["userId"]: u["userLabel"] for u in users_with_test}
        print(f"Evaluating for {len(user_ids)} users who have test data.")

        detailed_results = []

        sum_hit = 0.0
        sum_recall = 0.0
        sum_precision = 0.0
        sum_ndcg = 0.0
        total_users = 0

        for user_id in tqdm(user_ids):
            # Ground truth from TEST_RATED
            gt_query = """
                MATCH (u:User)
                WHERE elementId(u) = $userEid
                MATCH (u)-[:TEST_RATED]->(m:Movie)
                RETURN coalesce(toString(m.id), m.title, elementId(m)) AS movieId
            """
            gt_list = [r["movieId"] for r in self.graph.query(gt_query, {"userEid": user_id})]
            ground_truth = set(gt_list)

            if not ground_truth:
                continue

            # recommendations
            try:
                recommendations = self.get_recommendations_by_model(user_id, model_name=model_name)
            except Exception as e:
                print(f"Error getting recommendations for user {user_id}: {e}")
                continue

            # metrics
            hit_val = self._hit_at_k(recommendations, ground_truth, self.k)
            recall_val = self._recall_at_k(recommendations, ground_truth, self.k)
            precision_val = self._precision_at_k(recommendations, ground_truth, self.k)
            ndcg_val = self._ndcg_at_k(recommendations, ground_truth, self.k)

            sum_hit += hit_val
            sum_recall += recall_val
            sum_precision += precision_val
            sum_ndcg += ndcg_val
            total_users += 1

            hit_set = set(recommendations[: self.k]).intersection(ground_truth)

            detailed_results.append(
                {
                    "user_id": user_id,
                    "user_label": user_labels.get(user_id, user_id),
                    "model": model_name,
                    "k": self.k,
                    "ground_truth_count": len(ground_truth),
                    "recommended_count": len(recommendations),
                    "relevant_found": len(hit_set),
                    "hit": int(hit_val),
                    "recall": round(recall_val, 4),
                    "precision": round(precision_val, 4),
                    "ndcg": round(ndcg_val, 4),
                    "ground_truth_movies": str(list(ground_truth)),
                    "recommended_movies": str(recommendations),
                }
            )

        # aggregates
        if total_users == 0:
            metrics = {
                "Model": model_name,
                "K": self.k,
                "Users": 0,
                "Hit@K": 0.0,
                "Recall@K": 0.0,
                "Precision@K": 0.0,
                "NDCG@K": 0.0,
            }
            print("No users evaluated (total_users=0).")
            return metrics, pd.DataFrame(detailed_results)

        hit_rate = sum_hit / total_users
        avg_recall = sum_recall / total_users
        avg_precision = sum_precision / total_users
        avg_ndcg = sum_ndcg / total_users

        metrics = {
            "Model": model_name,
            "K": self.k,
            "Users": total_users,
            "Hit@K": round(hit_rate, 4),
            "Recall@K": round(avg_recall, 4),
            "Precision@K": round(avg_precision, 4),
            "NDCG@K": round(avg_ndcg, 4),
        }

        print("\n" + "=" * 40)
        print(f"Evaluation Results ({model_name}) (Top {self.k})")
        print("=" * 40)
        print(f"Total Users Evaluated: {total_users}")
        print(f"Hit@{self.k}:       {hit_rate:.4f}")
        print(f"Recall@{self.k}:    {avg_recall:.4f}")
        print(f"Precision@{self.k}: {avg_precision:.4f}")
        print(f"NDCG@{self.k}:      {avg_ndcg:.4f}")
        print("=" * 40)

        df_results = pd.DataFrame(detailed_results)

        if save_csv and not df_results.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_details_{model_name}_k{self.k}_{timestamp}.csv"
            df_results.to_csv(filename, index=False)
            print(f"\n[Documentation] Detailed results saved to: {filename}")

        return metrics, df_results

    # ---------------------------------------------------------------------
    # 7) Convenience: run all three models and produce a summary table
    # ---------------------------------------------------------------------
    def run_all_models(self, k=10, save_csv=True):
        """
        Runs evaluation for:
        - Random
        - Popularity
        - Ours (Graph-CF)
        Returns summary DataFrame.
        """
        summaries = []
        for model in ["random", "popularity", "ours"]:
            metrics, _ = self.run_evaluation(k=k, model_name=model, save_csv=save_csv)
            if metrics:
                summaries.append(metrics)
        return pd.DataFrame(summaries)


if __name__ == "__main__":
    evaluator = MovieRecommenderEvaluator(graph)

    try:
        evaluator.check_data()

        # 1) Create test split inside Neo4j
        evaluator.prepare_test_set(test_ratio=0.2)

        # 2) Run all models @10 and print/return summary
        summary = evaluator.run_all_models(k=10, save_csv=True)
        print("\nSummary Table:")
        print(summary.to_string(index=False))

    finally:
        # Restore graph
        evaluator.cleanup()
