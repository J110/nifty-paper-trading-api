"""
Load trained model and generate predictions.
Same model used by all versions — only the signal mapping differs.
"""

import os
import logging
import numpy as np
import joblib

logger = logging.getLogger(__name__)


class ModelRunner:
    """
    Load trained GradientBoosting/LightGBM model and generate predictions.
    The model predicts max_drawdown_30d from feature vectors.
    """

    def __init__(self, model_path: str, scaler_path: str = None,
                 feature_names_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_names = None

        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")

        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")

        if feature_names_path and os.path.exists(feature_names_path):
            self.feature_names = joblib.load(feature_names_path)
            logger.info(f"Loaded feature names ({len(self.feature_names)} features)")
        elif self.model and hasattr(self.model, 'feature_names_in_'):
            self.feature_names = list(self.model.feature_names_in_)
        elif self.model and hasattr(self.model, 'feature_name'):
            # LightGBM
            self.feature_names = self.model.feature_name()

    def predict(self, features: dict) -> float:
        """
        Predict max_drawdown_30d from feature dict.
        Returns predicted drawdown as negative decimal (e.g., -0.025 = -2.5%).
        """
        if self.model is None:
            logger.error("No model loaded — returning neutral prediction")
            return -0.02  # neutral default

        if self.feature_names:
            # Order features to match training
            feature_vector = np.array(
                [features.get(f, 0.0) for f in self.feature_names]
            ).reshape(1, -1)
        else:
            feature_vector = np.array(list(features.values())).reshape(1, -1)

        if self.scaler:
            feature_vector = self.scaler.transform(feature_vector)

        prediction = self.model.predict(feature_vector)[0]
        logger.info(f"Model prediction: {prediction:.4f} ({prediction*100:.2f}%)")
        return float(prediction)

    def get_feature_importances(self) -> dict:
        """Return feature importance dict for display."""
        if self.model is None or self.feature_names is None:
            return {}

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            return {}

        return dict(sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        ))
