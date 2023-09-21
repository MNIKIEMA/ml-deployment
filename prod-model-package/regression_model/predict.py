import typing as t

import numpy as np
import pandas as pd

class Predictor:
    
    def __init__(self, version: str, price_pipe: object,
                 config: object) -> None:
        self.version = version
        self.price_pipe = price_pipe
        self.config = config

    def make_prediction(self, input_data: t.Union[pd.DataFrame, dict],) -> dict:
        """Make a prediction using a saved model"""
        data = pd.DataFrame(input_data)
        validated_data, errors = validate_inputs(input_data=data)

        results = {"predictions": None, "version": _version, "errors": errors}
        if not errors:
            predictions = _price_pipe.predict(
                X=validated_data[config.model_config.features])
            
            results = {
                "predictions": [np.exp(pred) for pred in predictions],  # type: ignore
                "version": _version,
                "errors": errors,
            }

        return results