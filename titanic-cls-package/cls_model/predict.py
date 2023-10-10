import typing as t

import numpy as np
import pandas as pd

from cls_model import __version__ as _version
from cls_model.config.core import config
from cls_model.processing.data_manager import load_pipeline
from cls_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_titanic_pipe = load_pipeline(file_name=pipeline_file_name)

class Predictor:
    
    def __init__(self, version: str = _version, price_pipe: object = _titanic_pipe,
                 config: object = config) -> None:
        self.version = version
        self.price_pipe = price_pipe
        self.config = config

    def make_prediction(self, input_data: t.Union[pd.DataFrame, dict],) -> dict:
        """Make a prediction using a saved model"""
        data = pd.DataFrame(input_data)
        validated_data, errors = validate_inputs(input_data=data)

        results = {"predictions": None, "version": _version, "errors": errors}
        if not errors:
            predictions = _titanic_pipe.predict(
                X=validated_data[config.model_config.features])
            
            results = {
                "predictions": predictions,  # type: ignore
                "version": _version,
                "errors": errors,
            }

        return results