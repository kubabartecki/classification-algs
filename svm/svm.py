import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

class MySVM:
    def __init__(self):
        self.clf_svm = None
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None

        # visualization 2D
        self.pca = None
        self.pca_train_scaled = None
        self.X_train_pca = None

        # for cross validation
        self.param_grid = [
            {
                'C': np.linspace(0.01,50,5),
                'gamma': np.linspace(0.01,50,5),
                'degree': [9],
                'coef0': [6.266363636363636],
                'kernel': ['poly'],
                'max_iter': [28]
            }
        ]

        self.optimal_params = GridSearchCV(
            SVC(),
            self.param_grid,
            cv=5,
            scoring='accuracy',
            verbose=2
        )

    def prepare_svm(self):
        self.clf_svm = SVC(kernel='rbf', random_state=42, max_iter=35, C=20, gamma=0.002)
        self.clf_svm.fit(self.X_train, self.y_train)

    def prepare_dataset(self):
        self.X_train = pd.read_csv('dataset/X_train.csv')
        self.y_train = pd.read_csv('dataset/y_train.csv')
        self.X_test = pd.read_csv('dataset/X_test.csv')
        self.y_test = pd.read_csv('dataset/y_test.csv')
    
    def prepare(self):
        self.prepare_dataset()
        self.prepare_svm()

    def perform_test_dataset(self):
        # classification
        predictions = self.clf_svm.predict(self.X_test)
        cm = confusion_matrix(self.y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=[num + 1 for num in range(5)])
        disp.plot()
        plt.show()

    def plot_3d_cv(self):
        gamma = np.linspace(1, 40, 20)
        C = np.linspace(0.01, 4, 20)
        tests_number = len(self.y_test)
        ratio = []
        best_result = 0
        best_result_params =[]
        for g in gamma:
            success = []
            for c in C:
                self.clf_svm = SVC(kernel='rbf', random_state=42, C=c, gamma=g)
                self.clf_svm.fit(self.X_train, self.y_train)
                pred = self.clf_svm.predict(self.X_test)
                is_correct = [not(a ^ b) for a, b in zip(self.y_test.values, pred)]
                value = sum(is_correct) / tests_number
                success.append(value)

                if value > best_result:
                    best_result = value
                    best_result_params = [(g, c)]
                elif value == best_result:
                    best_result_params.append((g, c))
                print(g,c, value)
            ratio.append(success)

        print(best_result, best_result_params)
        x = np.array(gamma)
        y = np.array(C)
        X, Y = np.meshgrid(x, y)
        Z = np.array(ratio)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_xlabel('gamma')
        ax.set_ylabel('C')
        ax.set_zlabel('Efficiency')
        ax.view_init(60, 30)
        plt.show()

if __name__ == '__main__':
    mySVM = MySVM()
    mySVM.prepare()
    mySVM.perform_test_dataset()
    # mySVM.plot_3d_cv()
