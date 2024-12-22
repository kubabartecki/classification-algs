import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_model():
   
    X_train = pd.read_csv('dataset/X_train.csv')
    y_train = pd.read_csv('dataset/y_train.csv')
    X_test = pd.read_csv('dataset/X_test.csv')
    y_test = pd.read_csv('dataset/y_test.csv')

    # use multi-layer perceptron classifier
    mlp = MLPClassifier(hidden_layer_sizes=(12, 7, 5), max_iter=1000, random_state=42)
    
    # training the model
    mlp.fit(X_train, y_train)

    # predict the response for test dataset
    y_pred = mlp.predict(X_test)

    return y_test, y_pred

def evaluate_model(y_test, y_pred):
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
