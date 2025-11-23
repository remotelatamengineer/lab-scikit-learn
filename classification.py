import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def test_classification():
    print("--- Classification Test ---")
    # Load dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
    print(classification_report(y_test, y_pred_rf, target_names=iris.target_names))
    
    # Train SVM
    print("Training SVM...")
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}")
    print(classification_report(y_test, y_pred_svm, target_names=iris.target_names))

if __name__ == "__main__":
    test_classification()
