from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC

def test_model_selection():
    print("--- Model Selection Test ---")
    # Load dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Cross Validation
    print("Running Cross Validation...")
    clf = SVC(kernel='linear', C=1, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean score: {scores.mean():.2f}")
    
    # Grid Search
    print("Running Grid Search...")
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Best Estimator: {grid.best_estimator_}")

if __name__ == "__main__":
    test_model_selection()
