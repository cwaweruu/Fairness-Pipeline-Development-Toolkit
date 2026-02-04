import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config_models import DataConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and train/test splitting."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def load_and_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load dataset and create train/test split.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("=" * 80)
        logger.info("DATA LOADING")
        logger.info("=" * 80)
        
        if self.config.path is None:
            raise ValueError("data.path must be specified in configuration")
        
        data_path = Path(self.config.path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data from {data_path}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Dataset columns: {df.columns.tolist()}")
        
        target_col = self.config.target_column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Protected attributes: {self.config.protected_attributes}")
        
        return X_train, X_test, y_train, y_test
