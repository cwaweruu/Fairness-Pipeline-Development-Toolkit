import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import mlflow
import mlflow.sklearn

from src.config_models import MLflowConfig

logger = logging.getLogger(__name__)


class MLflowLogger:
    """Handles MLflow experiment tracking and artifact logging."""
    
    def __init__(self, config: MLflowConfig):
        self.config = config
        self.run_id = None
    
    def log_experiment(
        self,
        config_path: Path,
        baseline_metrics: Dict[str, float],
        final_metrics: Dict[str, float],
        model: Any,
        training_method: str,
        preprocessing_transformers: list,
        primary_metric: str
    ) -> None:
        """
        Log complete experiment to MLflow.
        
        Args:
            config_path: Path to configuration file
            baseline_metrics: Baseline fairness metrics
            final_metrics: Final performance and fairness metrics
            model: Trained model
            training_method: Training method used
            preprocessing_transformers: List of preprocessing transformers
            primary_metric: Primary fairness metric
        """
        logger.info("=" * 80)
        logger.info("STEP 4: MLFLOW LOGGING")
        logger.info("=" * 80)
        
        mlflow.set_experiment(self.config.experiment_name)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.config.run_name_prefix}_{timestamp}"
        
        with mlflow.start_run(run_name=run_name):
            self.run_id = mlflow.active_run().info.run_id
            
            mlflow.log_artifact(str(config_path), artifact_path="config")
            
            mlflow.log_param("preprocessing_transformers", preprocessing_transformers)
            mlflow.log_param("training_method", training_method)
            mlflow.log_param("primary_fairness_metric", primary_metric)
            
            for key, value in self.config.tags.items():
                mlflow.set_tag(key, value)
            
            for metric, value in baseline_metrics.items():
                mlflow.log_metric(f"baseline_{metric}", value)
            
            for metric, value in final_metrics.items():
                mlflow.log_metric(f"final_{metric}", value)
            
            if primary_metric in baseline_metrics and primary_metric in final_metrics:
                improvement = baseline_metrics[primary_metric] - final_metrics[primary_metric]
                mlflow.log_metric("fairness_improvement", improvement)
            
            if model is not None:
                mlflow.sklearn.log_model(model, "model")
            
            logger.info("Logged to MLflow successfully")
            logger.info(f"  Experiment: {self.config.experiment_name}")
            logger.info(f"  Run ID: {self.run_id}")
