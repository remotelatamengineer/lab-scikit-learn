import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

def test_preprocessing():
    print("--- Preprocessing Test ---")
    data = np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]])
    print(f"Original Data:\n{data}")
    
    # StandardScaler
    print("StandardScaler:")
    scaler = StandardScaler()
    print(scaler.fit_transform(data))
    
    # MinMaxScaler
    print("MinMaxScaler:")
    min_max = MinMaxScaler()
    print(min_max.fit_transform(data))
    
    # OneHotEncoder
    print("OneHotEncoder:")
    enc = OneHotEncoder(handle_unknown='ignore')
    X = [['Male', 1], ['Female', 3], ['Female', 2]]
    enc.fit(X)
    print(f"Categories: {enc.categories_}")
    print(f"Transform [['Female', 1], ['Male', 4]]:\n{enc.transform([['Female', 1], ['Male', 4]]).toarray()}")

if __name__ == "__main__":
    test_preprocessing()
