from typing import Optional, List, Any
from pydantic import BaseModel

import pandas as pd
import numpy as np

from evaluate.schemas.metrics import EvaluationMetric
from evaluate.utils.validators import validate_classifier


class Truth(BaseModel):

    name: str
    y_pred: np.array

    class Config:
        arbitrary_types_allowed = True


class Evaluator(BaseModel):

    test_df: pd.DataFrame
    truths: List[Truth]
    estimator: Any         # TODO: type check this better than Any
    metrics: List[EvaluationMetric]
    slices: Optional[List[str]]
    
    class Config:
        arbitrary_types_allowed = True

    def pre_validate(self) -> None:

        classif = validate_classifier(self.estimator)
        if classif: 
            model_type = 'classifier'
        else:
            # NB just covering corner case for non-classifier as regressor, would build this out to be complete
            model_type = 'regressor'

        for y in self.evaluations:
            if model_type not in y.metric.allowed_models:
                raise ValueError(f"{model_type} is not a valid model type for {y.metric.name} metric")

    def evaluate(self) -> pd.DataFrame:

        # before committing compute resources, validate metrics against model type
        self.pre_validate()

        # compute predictions TODO: check valid sklearn model
        y_pred = self.estimator.predict(self.test_df.to_numpy())
        for truth in self.truths: 
            self.test_df[truth.name] = truth.y_pred    

        evaluations = [
            {
                "eval_name": f"{x.name}-{truth.name}", 
                "metric_fn": x.metric_fn, 
                "y_true": truth
            } for x in self.metrics for truth in self.truths
        ]

        # compute metrics
        evals = self.test_df.groupby(self.slices).apply(
            lambda x: pd.Series(
                data=[e["metric_fn"](y_pred, e["y_true"]) for e in evaluations], 
                index=[ e["eval_name"] for e in evaluations]
            )
        )

        return evals