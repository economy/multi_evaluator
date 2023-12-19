import numpy as np

def specificity(y_pred: np.array, y_true: np.array) -> np.float64:
    # true negative rate
    tnr = ((y_pred == y_true) & (y_true != 1)).sum() / (y_true != 1).sum()

    return tnr