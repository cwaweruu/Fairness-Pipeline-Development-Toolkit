import logging
from typing import Dict

import pandas as pd
import numpy as np

from src.config_models import FairnessConfig

logger = logging.getLogger(__name__)


class BaselineMeasurement:
    """Performs baseline fairness measurements on data."""
    
    def __init__(self, config: FairnessConfig, protected_attributes: list):
        self.config = config
        self.protected_attributes = protected_attributes
    
    def measure(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Calculate baseline fairness metrics.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary of fairness metrics
        """
        logger.info("=" * 80)
        logger.info("STEP 1: BASELINE FAIRNESS MEASUREMENT")
        logger.info("=" * 80)
        
        metrics = {}
        
        if not self.protected_attributes:
            logger.warning("No protected attributes specified")
            return metrics
        
        attr = self.protected_attributes[0]
        if attr not in X.columns:
            logger.warning(f"Protected attribute '{attr}' not found in features")
            return metrics
        
        groups = X[attr].unique()
        selection_rates = []
        
        logger.info(f"\nBaseline Analysis by {attr}:")
        for group in groups:
            mask = X[attr] == group
            rate = y[mask].mean() if mask.sum() > 0 else 0
            selection_rates.append(rate)
            logger.info(f"  {attr}={group}: Selection rate = {rate:.4f}")
        
        if len(selection_rates) >= 2:
            dp_diff = max(selection_rates) - min(selection_rates)
            metrics['demographic_parity_difference'] = dp_diff
            logger.info(f"\nBaseline Demographic Parity Difference: {dp_diff:.4f}")
        
        return metrics
