from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def test_dim_reduction():
    print("--- Dimensionality Reduction Test ---")
    # Load digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target
    print(f"Original shape: {X.shape}")
    
    # PCA
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print(f"Reduced shape: {X_pca.shape}")
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

if __name__ == "__main__":
    test_dim_reduction()
