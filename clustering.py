import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def test_clustering():
    print("--- Clustering Test ---")
    # Generate synthetic data
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    
    # K-Means Clustering
    print("Running K-Means...")
    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    
    # Silhouette Score
    score = silhouette_score(X, y_kmeans)
    print(f"Silhouette Score: {score:.2f}")
    print(f"Cluster Centers:\n{kmeans.cluster_centers_}")

if __name__ == "__main__":
    test_clustering()
