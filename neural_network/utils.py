import numpy as np

def check_accuracy(y_pred, y_train):
    assert y_pred.shape == y_train.shape, "Arrays must have the same shape"
    matches = np.sum(y_pred == y_train)
    total_elements = y_pred.size
    match_percentage = (matches / total_elements) * 100
    
    return match_percentage
