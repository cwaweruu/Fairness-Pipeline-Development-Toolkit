import logging
from typing import Dict

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config_models import FairnessConfig

logger = logging.getLogger(__name__)


class FinalValidator:
    """Validates model fairness and performance."""
    
    def __init__(self, config: FairnessConfig, protected_attributes: list):
        self.config = config
        self.protected_attributes = protected_attributes
    
    def validate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Validate model performance and fairness.
        
        Args:
            X_test: Test features
            y_test: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of performance and fairness metrics
        """
        logger.info("=" * 80)
        logger.info("STEP 3: FINAL VALIDATION")
        logger.info("=" * 80)
        
        metrics = {}
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        metrics['accuracy'] = accuracy
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        
        logger.info("\nPerformance Metrics:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")
        
        if self.protected_attributes:
            fairness_metrics = self._calculate_fairness(X_test, y_pred)
            metrics.update(fairness_metrics)
        
        return metrics
    
    def _calculate_fairness(self, X_test: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate fairness metrics."""
        metrics = {}
        
        attr = self.protected_attributes[0]
        if attr not in X_test.columns:
            logger.warning(f"Protected attribute '{attr}' not found")
            return metrics
        
        groups = X_test[attr].unique()
        selection_rates = []
        
        logger.info(f"\nFairness Analysis by {attr}:")
        for group in groups:
            mask = X_test[attr] == group
            rate = y_pred[mask].mean() if mask.sum() > 0 else 0
            selection_rates.append(rate)
            logger.info(f"  {attr}={group}: Selection rate = {rate:.4f}")
        
        if len(selection_rates) >= 2:
            dp_diff = max(selection_rates) - min(selection_rates)
            metrics['demographic_parity_difference'] = dp_diff
            logger.info(f"\nDemographic Parity Difference: {dp_diff:.4f}")
        
        return metrics
    
    def generate_comparison(self, baseline_metrics: Dict[str, float], final_metrics: Dict[str, float]) -> None:
        """Generate before/after comparison report."""
        logger.info("\n" + "=" * 80)
        logger.info("FAIRNESS IMPROVEMENT REPORT")
        logger.info("=" * 80)
        
        primary_metric = self.config.primary_metric
        threshold = self.config.threshold
        
        baseline_value = baseline_metrics.get(primary_metric, np.nan)
        final_value = final_metrics.get(primary_metric, np.nan)
        
        if not np.isnan(baseline_value) and not np.isnan(final_value):
            improvement = baseline_value - final_value
            improvement_pct = (improvement / baseline_value * 100) if baseline_value != 0 else 0
            
            logger.info(f"\nPrimary Fairness Metric: {primary_metric}")
            logger.info(f"  Baseline:    {baseline_value:.4f}")
            logger.info(f"  Final:       {final_value:.4f}")
            logger.info(f"  Improvement: {improvement:.4f} ({improvement_pct:+.1f}%)")
            logger.info(f"  Threshold:   {threshold:.4f}")
            
            if final_value <= threshold:
                logger.info("  Status: PASSED (within threshold)")
            else:
                logger.info("  Status: FAILED (exceeds threshold)")
        
        logger.info("=" * 80 + "\n")
