import numpy as np

from schemas.metrics import EvaluationMetric

def sens(y_pred: np.array, y_true: np.array) -> np.float64:
    # true positive rate
    tpr = ((y_pred == y_true) & (y_true == 1)).sum() / (y_true == 1).sum()
    
    return tpr

sensitivity = EvaluationMetric(
    name = "sensitivity",
    metric_fn = sens,
    allowed_models = ["classifier"]
)