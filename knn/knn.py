import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
import pandas as pd

def knn_predict(X_train, y_train, X_test, k=3):
    # ensure the inputs are 2D arrays
    if X_train.ndim == 1:
        X_train = np.expand_dims(X_train, axis=1)
    if X_test.ndim == 1:
        X_test = np.expand_dims(X_test, axis=1)

    predictions = []
    for test_point in X_test:
        distances = np.linalg.norm(X_train - test_point, axis=1)
        k_nearest_neighbors = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_nearest_neighbors]
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])
    return np.array(predictions)


def calculate_precision(y_test_path, y_pred):
    y_test = pd.read_csv(y_test_path).values.ravel()
    return accuracy_score(y_test, y_pred)


def test_K_param_knn(X, y, k=3):
    for k in range(1, 10):
        knn = KNNClassifier(k)
        knn.fit(X, y)
        y_pred = knn.predict(X_test_path)
        precision = calculate_precision(y_test_path, y_pred)
        print(f"Accuracy for k={k}: {precision:.2f}")

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train_path, y_train_path):
        self.X_train = pd.read_csv(X_train_path).values
        self.y_train = pd.read_csv(y_train_path).values.ravel()

    def predict(self, X_test_path):
        if self.X_train is None or self.y_train is None:
            raise ValueError("The classifier has not been fitted with training data.")
        X_test = pd.read_csv(X_test_path).values 
        return knn_predict(self.X_train, self.y_train, X_test, self.k)

if __name__ == '__main__':
    X_train_path = 'dataset/X_train.csv'
    y_train_path = 'dataset/y_train.csv'
    X_test_path = 'dataset/X_test.csv'
    y_test_path = 'dataset/y_test.csv'

    knn = KNNClassifier(k=4)
    knn.fit(X_train_path, y_train_path)

    y_pred = knn.predict(X_test_path)
    precision = calculate_precision(y_test_path, y_pred)
    print(f"Accuracy: {precision:.2f}")

    test_K_param_knn(X_train_path, y_train_path, k=4)
