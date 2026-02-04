import logging
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config_models import PreprocessingConfig

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Applies data preprocessing transformations."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.scaler = StandardScaler()
    
    def transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        protected_attributes: list
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply preprocessing transformations to data.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            protected_attributes: List of protected attribute names
            
        Returns:
            Tuple of (X_train_transformed, X_test_transformed)
        """
        logger.info("=" * 80)
        logger.info("STEP 2A: PREPROCESSING")
        logger.info("=" * 80)
        
        if not self.config.enabled:
            logger.info("Preprocessing disabled in configuration")
            return X_train, X_test
        
        logger.info(f"Applying transformers: {self.config.transformers}")
        logger.info(f"Repair level: {self.config.repair_level}")
        
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        numeric_cols = X_train_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = X_train_processed.select_dtypes(exclude=[np.number]).columns
        
        X_train_processed[numeric_cols] = self.scaler.fit_transform(
            X_train_processed[numeric_cols]
        )
        X_test_processed[numeric_cols] = self.scaler.transform(X_test_processed[numeric_cols])
        
        logger.info(f"Scaled {len(numeric_cols)} numeric features")
        logger.info(f"Preserved {len(categorical_cols)} categorical features")
        logger.info("Preprocessing complete")
        
        return X_train_processed, X_test_processed
