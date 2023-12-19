import numpy as np

def sensitivity(y_pred: np.array, y_true: np.array) -> np.float64:
    # true positive rate
    tpr = ((y_pred == y_true) & (y_true == 1)).sum() / (y_true == 1).sum()
    
    return tpr