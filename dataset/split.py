import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset():
    df = pd.read_csv('dataset/processed_dataset.csv')

    if 'User ID' in df.columns:
        df = df.drop(columns=['User ID'])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training dataset size:", X_train.shape)
    print("Testing dataset size:", X_test.shape)
    
    X_train.to_csv("dataset/X_train.csv", index=False)
    X_test.to_csv("dataset/X_test.csv", index=False)
    y_train.to_csv("dataset/y_train.csv", index=False)
    y_test.to_csv("dataset/y_test.csv", index=False)
