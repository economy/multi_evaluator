from enum import Enum
from typing import Any, Optional, List, Callable
from pydantic import BaseModel

import numpy as np


class ModelType(str, Enum):
    classifier = 'classifier'
    regressor = 'regressor'
    ranker = 'ranker'

class EvaluationMetric(BaseModel):

    name: str
    metric_fn: Callable
    allowed_models: List[ModelType]

    def compute(self, y_pred: np.array, y_true: np.array) -> np.float64:
        # must take arrays as input, return float value


        statistic = self.metric_fn(y_pred, y_true)

        return statistic