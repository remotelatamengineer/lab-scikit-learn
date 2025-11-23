import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

def test_regression():
    print("--- Regression Test ---")
    # Load dataset
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print(f"Linear Regression MSE: {mean_squared_error(y_test, y_pred_lr):.2f}")
    print(f"Linear Regression R2: {r2_score(y_test, y_pred_lr):.2f}")
    
    # Ridge Regression
    print("Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    print(f"Ridge Regression MSE: {mean_squared_error(y_test, y_pred_ridge):.2f}")
    print(f"Ridge Regression R2: {r2_score(y_test, y_pred_ridge):.2f}")

if __name__ == "__main__":
    test_regression()
