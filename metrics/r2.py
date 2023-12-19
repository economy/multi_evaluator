import numpy as np

from schemas.metrics import EvaluationMetric


def r_square(y_pred: np.array, y_true: np.array) -> np.float64:
    # doesn't matter if inputs are continuous, [-1,1], [0,1], or otherwise
    sse = ((y_true - y_pred) ** 2).sum(axis=0)
    sst = ((y_true - np.mean(y_true)) ** 2).sum(axis=0)
    r2 = 1 - (sse / sst)

    return r2

    