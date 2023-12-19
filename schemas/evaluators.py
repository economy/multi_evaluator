from typing import Optional, List, Any
from pydantic import BaseModel

import pandas as pd
import numpy as np

from schemas.metrics import EvaluationMetric
from utils.validators import validate_classifier

class Evaluator(BaseModel):

    test_df = pd.DataFrame
    y_true = np.array
    estimator = Any
    evaluation_metrics = List[EvaluationMetric]
    slices = Optional[List[str]]

    def pre_validate(self) -> None:

        classif = validate_classifier(self.estimator)
        if classif: 
            model_type = 'classifier'
        else:
            # NB just covering corner case for non-classifier as regressor, would build this out to be complete
            model_type = 'regressor'

        for y in self.evaluation_metrics:
            if model_type not in y.allowed_models:
                raise ValueError(f"{model_type} is not a valid model type for {y.name} metric")

    def evaluate(self) -> pd.DataFrame:

        # before committing compute resources, validate metrics against model type
        self.pre_validate()

        # compute predictions TODO: check valid sklearn model
        y_hat = self.estimator.predict()
        self.test_df['y_true'] = self.y_true
        self.test_df['y_pred'] = y_hat

        # compute metrics
        evals = self.test_df.groupby(self.slices).apply(lambda x: pd.Series(data=[y(x) for y in self.evaluation_metrics], index=[y.name for y in self.evaluation_metrics]))

        return evals