import numpy as np

from schemas.metrics import EvaluationMetric


def spec(y_pred: np.array, y_true: np.array) -> np.float64:
    # true negative rate
    tnr = ((y_pred == y_true) & (y_true != 1)).sum() / (y_true != 1).sum()

    return tnr

specificity = EvaluationMetric(
    name = "specificity",
    metric_fn = spec,
    allowed_models = ["classifier"]
)
