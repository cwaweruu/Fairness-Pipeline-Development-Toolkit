import logging
from typing import Any

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from src.config_models import TrainingConfig

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains machine learning models with fairness constraints."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.label_encoders = {}
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        protected_attributes: list
    ) -> Any:
        """
        Train a fair model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            protected_attributes: List of protected attribute names
            
        Returns:
            Trained model
        """
        logger.info("=" * 80)
        logger.info("STEP 2B: MODEL TRAINING")
        logger.info("=" * 80)
        
        logger.info(f"Training method: {self.config.method}")
        logger.info(f"Fairness constraint: {self.config.constraint}")
        
        X_train_encoded = self._encode_features(X_train, fit=True)
        
        self.model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=42
        )
        
        logger.info(f"Training with {X_train_encoded.shape[0]} samples")
        logger.info(f"Features: {X_train_encoded.shape[1]}")
        
        self.model.fit(X_train_encoded, y_train)
        
        logger.info("Model training complete")
        
        return self.model
    
    def _encode_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Encode categorical features as numeric values."""
        X_encoded = X.copy()
        
        for col in X_encoded.select_dtypes(exclude=[np.number]).columns:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col].astype(str))
            else:
                if col in self.label_encoders:
                    X_encoded[col] = self.label_encoders[col].transform(X_encoded[col].astype(str))
        
        return X_encoded.values
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with trained model."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X_encoded = self._encode_features(X, fit=False)
        return self.model.predict(X_encoded)
