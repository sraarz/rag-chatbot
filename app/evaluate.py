import pandas as pd
import random
from tqdm import tqdm
from graph import graph
import os
from datetime import datetime

class MovieRecommenderEvaluator:
    def __init__(self, graph):
        self.graph = graph
        self.driver = graph._driver
        # ratings_csv_path حذف شد چون دیگر از فایل استفاده نمی‌کنیم
        self.k = 10  # Top K recommendations to evaluate

    def check_data(self):
        """
        Checks if data exists in the Neo4j graph instead of loading from CSV.
        """
        # بررسی وجود رابطه RATED برای اطمینان از داده‌ها
        check_query = "MATCH ()-[r:RATED]->() RETURN count(r) > 0 as exists"
        result = self.graph.query(check_query)
        
        if result and result[0]['exists']:
            print("Data already exists in the graph. Skipping CSV load.")
        else:
            print("WARNING: No RATED relationships found in the graph!")
            print("Please make sure you have loaded your movie data into Neo4j first.")

        try:
            self.graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
            self.graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE")
            print("Constraints verified.")
        except Exception as e:
            print(f"Note: Could not create constraints (they might exist): {e}")

    def prepare_test_set(self, test_ratio=0.2):
        """
        Splits the data into Train and Test sets based on existing data in Neo4j.
        """
        print(f"Preparing test set (splitting {test_ratio*100}% of data)...")
        
        self.cleanup() # Restore state first

        # Assign random number to relationships
        self.graph.query("""
            MATCH (u:User)-[r:RATED]->(m:Movie)
            SET r.random = rand()
        """)

        # Split logic: تبدیل بخشی از رابطه‌ها به TEST_RATED
        # برای کاربرانی که تعداد کمی امتیاز دارند، حداقل 1 نمونه تست نگه می‌داریم
        # (به شرطی که بیشتر از 1 رأی داشته باشند تا داده آموزشی هم باقی بماند)
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

        stats = self.graph.query("""
            MATCH ()-[tr:TEST_RATED]->()
            RETURN count(tr) AS test_count
        """)
        print(f"Created TEST_RATED relationships: {stats[0]['test_count'] if stats else 0}")
        
        # حذف ویژگی موقت random
        self.graph.query("MATCH ()-[r:RATED]->() REMOVE r.random")
        print("Test set prepared. Original ratings masked.")

    def get_recommendations(self, user_id):
        """
        Generates recommendations using Collaborative Filtering.
        این قسمت بدون تغییر است چون مستقیماً از گراف کوئری می‌کشد.
        """
        query = """
        MATCH (target:User)
        WHERE elementId(target) = $userEid
        MATCH (target)-[:RATED]->(m:Movie)<-[:RATED]-(other:User)
        WITH target, other, count(m) as intersection
        WHERE intersection > 0
        MATCH (other)-[r:RATED]->(rec:Movie)
        WHERE NOT (target)-[:RATED]->(rec)
        WITH rec, count(r) as freq
        RETURN coalesce(toString(rec.id), rec.title, elementId(rec)) AS movieId
        ORDER BY freq DESC
        LIMIT $k
        """
        results = self.graph.query(query, {"userEid": user_id, "k": self.k})
        return [r['movieId'] for r in results]

    def run_evaluation(self, k=10):
        self.k = k
        print(f"\nStarting Evaluation @ {k}...")
        
        users_with_test = self.graph.query("""
            MATCH (u:User)-[:TEST_RATED]->()
            RETURN DISTINCT elementId(u) AS userId, coalesce(toString(u.id), u.name, elementId(u)) AS userLabel
        """)
        
        if not users_with_test:
            print("No test data found. Did you run prepare_test_set?")
            return

        user_ids = [u['userId'] for u in users_with_test]
        user_labels = {u['userId']: u['userLabel'] for u in users_with_test}
        print(f"Evaluating for {len(user_ids)} users who have test data.")

        detailed_results = []
        hits = 0
        recall_sum = 0
        total_users = 0

        for user_id in tqdm(user_ids):
            # 1. Get Ground Truth (فیلم‌هایی که در تست جدا شده‌اند)
            gt_query = """
                MATCH (u:User)
                WHERE elementId(u) = $userEid
                MATCH (u)-[:TEST_RATED]->(m:Movie)
                RETURN coalesce(toString(m.id), m.title, elementId(m)) AS movieId
            """
            ground_truth = set([r['movieId'] for r in self.graph.query(gt_query, {"userEid": user_id})])
            
            if not ground_truth:
                continue

            # 2. Get Recommendations
            try:
                recommendations = self.get_recommendations(user_id)
            except Exception as e:
                print(f"Error getting recommendations for user {user_id}: {e}")
                continue

            # 3. Calculate Metrics
            rec_set = set(recommendations)
            hit_set = rec_set.intersection(ground_truth)
            
            is_hit = 1 if len(hit_set) > 0 else 0
            recall_val = len(hit_set) / len(ground_truth)
            
            if is_hit:
                hits += 1
            
            recall_sum += recall_val
            total_users += 1

            detailed_results.append({
                "user_id": user_id,
                "user_label": user_labels.get(user_id, user_id),
                "ground_truth_count": len(ground_truth),
                "recommended_count": len(recommendations),
                "relevant_found": len(hit_set),
                "hit": is_hit,
                "recall": round(recall_val, 4),
                "ground_truth_movies": str(list(ground_truth)),
                "recommended_movies": str(recommendations)
            })

        # Compute Aggregate Metrics
        hit_rate = hits / total_users if total_users > 0 else 0
        avg_recall = recall_sum / total_users if total_users > 0 else 0

        print("\n" + "="*30)
        print(f"Evaluation Results (Top {self.k})")
        print("="*30)
        print(f"Total Users Evaluated: {total_users}")
        print(f"Hit Rate: {hit_rate:.4f}")
        print(f"Recall:    {avg_recall:.4f}")
        print("="*30)

        # Save Results
        if detailed_results:
            df_results = pd.DataFrame(detailed_results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_details_k{self.k}_{timestamp}.csv"
            df_results.to_csv(filename, index=False)
            print(f"\n[Documentation] Detailed results saved to: {filename}")
        else:
            print("\nNo results to save.")

    def cleanup(self):
        print("Cleaning up test data and restoring original graph...")
        self.graph.query("""
            MATCH (u:User)-[tr:TEST_RATED]->(m:Movie)
            MERGE (u)-[r:RATED]->(m)
            SET r.rating = tr.rating, r.timestamp = tr.timestamp
            DELETE tr
        """)
        print("Graph restored.")

if __name__ == "__main__":
    # فرض بر این است که آبجکت graph از جای دیگری ایمپورت شده است
    evaluator = MovieRecommenderEvaluator(graph)
    
    try:
        # بررسی می‌کند که داده‌ها در دیتابیس وجود دارند
        evaluator.check_data() 
        
        # تقسیم داده‌ها به آموزش و تست (روی داده‌های موجود در دیتابیس)
        evaluator.prepare_test_set(test_ratio=0.2)
        
        # اجرای ارزیابی
        evaluator.run_evaluation(k=10)
    finally:
        # بازگرداندن وضعیت دیتابیس به حالت اولیه
        evaluator.cleanup()
