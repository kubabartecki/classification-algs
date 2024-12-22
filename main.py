from dataset.process import label_and_normalize
from dataset.split import split_dataset
from neural_network.mlp import train_model
from neural_network.mlp import evaluate_model

def prepare_dataset():
    label_and_normalize()
    split_dataset()

def run_neural_network():
    y_test, y_pred = train_model()
    evaluate_model(y_test, y_pred)

if __name__ == '__main__':
    prepare_dataset()
    run_neural_network()
