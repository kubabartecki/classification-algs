import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from utils import check_accuracy
import warnings

warnings.filterwarnings("ignore")


def train_model(layers: tuple, max_iter: int, learning_rate_init: float):
   
    X_train = pd.read_csv('dataset/X_train.csv')
    y_train = pd.read_csv('dataset/y_train.csv')
    X_test = pd.read_csv('dataset/X_test.csv')

    # use multi-layer perceptron classifier
    mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=max_iter,
                        learning_rate_init=learning_rate_init, random_state=42)
    
    # training the model
    mlp.fit(X_train, y_train)

    # predict the response for test dataset
    y_pred = mlp.predict(X_test)

    return y_pred

def evaluate_model(y_pred):
    y_test = pd.read_csv('dataset/y_test.csv')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def find_best_params():
    layers_list = [(5,), (5, 3), (12, 7, 5)]
    learning_rates = [0.001, 0.0001, 0.01]
    max_iters = [10, 25, 50, 100]
    
    y_test = pd.read_csv('dataset/y_test.csv')
    
    for max_iter in max_iters:
        results = []
        for lr in learning_rates:
            for layers in layers_list:
                y_pred = train_model(layers, max_iter, lr)
                score = check_accuracy(y_pred, y_test.values.flatten())
                results.append((layers, lr, max_iter, score.mean()))

        print(f"Max Iter: {max_iter}")
        best_result = max(results, key=lambda x: x[-1])
        print(best_result)

if __name__ == '__main__':
    find_best_params()
    evaluate_model(train_model((12, 7, 5), 50, 0.01))
