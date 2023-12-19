# Multi Evaluation Tool


### Example usage:
```
import numpy as np

from evaluate.metrics import r2, sensitivity, specificity
from evaluate.schemas.metrics import EvaluationMetric
from evaluate import Evaluator


def custom_comp(y_pred: np.array, y_true: np.array) -> np.float64:
    return y_pred.mean() - y_true.mean()

my_custom_metric = EvaluationMetric(
    name = "my custom metric",
    model_fn = custom_comp,
    allowed_models = ["classifier"]
)

evaluator = Evaluator(
    test_df = Xte,
    y_true = yte,
    estimator = sklearn_model,
    evaluation_metrics = [r2, sensitivity, specificity, my_custom_metric],
    slices = ['country']
)

evaluator.evaluate()
```