# lab-scikit-learn
Scikit-Learn Capabilities Test Project Walkthrough
I have created a Python project to test various capabilities of scikit-learn. The project includes separate scripts for different machine learning tasks.

Implemented Features
1. Classification (
classification.py
)
Goal: Test classification algorithms (Random Forest, SVM).
Dataset: Iris.
Result: Successfully trained models and printed accuracy/classification reports.
2. Regression (
regression.py
)
Goal: Test regression algorithms (Linear Regression, Ridge).
Dataset: Diabetes.
Result: Successfully trained models and printed MSE/R2 scores.
3. Clustering (
clustering.py
)
Goal: Test clustering (K-Means).
Dataset: Synthetic blobs.
Result: Performed clustering and calculated Silhouette Score.
4. Dimensionality Reduction (
dim_reduction.py
)
Goal: Test PCA.
Dataset: Digits.
Result: Reduced dimensions and printed explained variance ratio.
5. Model Selection (
model_selection.py
)
Goal: Test Cross-Validation and GridSearch.
Dataset: Iris.
Result: Performed 5-fold CV and Grid Search for SVM hyperparameters.
6. Preprocessing (
preprocessing.py
)
Goal: Test Scalers and Encoders.
Result: Demonstrated StandardScaler, MinMaxScaler, and OneHotEncoder.
Verification Results
All scripts were executed successfully in the virtual environment.

venv\Scripts\python classification.py
venv\Scripts\python regression.py
venv\Scripts\python clustering.py
venv\Scripts\python dim_reduction.py
venv\Scripts\python model_selection.py
venv\Scripts\python preprocessing.py

