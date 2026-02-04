import logging
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
from pydantic import ValidationError

from src.config_models import PipelineConfig
from src.data_loader import DataLoader
from src.baseline_measurement import BaselineMeasurement
from src.preprocessing import PreprocessingPipeline
from src.model_trainer import ModelTrainer
from src.final_validator import FinalValidator
from src.mlflow_logger import MLflowLogger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_run.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the complete fairness-aware ML pipeline."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = None
        self.baseline_metrics: Dict[str, float] = {}
        self.final_metrics: Dict[str, float] = {}
        self.model = None
        
        logger.info("Initializing Fairness Pipeline Orchestrator")
        logger.info(f"Configuration file: {self.config_path}")
    
    def load_configuration(self) -> None:
        """Load and validate pipeline configuration."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            self.config = PipelineConfig(**config_dict)
            logger.info("Configuration loaded and validated successfully")
            
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def run(self) -> None:
        """Execute the complete fairness pipeline."""
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STARTING FAIRNESS PIPELINE EXECUTION")
            logger.info("=" * 80 + "\n")
            
            self.load_configuration()
            
            data_loader = DataLoader(self.config.data)
            X_train, X_test, y_train, y_test = data_loader.load_and_split()
            
            baseline = BaselineMeasurement(
                self.config.fairness,
                self.config.data.protected_attributes
            )
            self.baseline_metrics = baseline.measure(X_train, y_train)
            
            preprocessor = PreprocessingPipeline(self.config.preprocessing)
            X_train_processed, X_test_processed = preprocessor.transform(
                X_train,
                X_test,
                y_train,
                self.config.data.protected_attributes
            )
            
            trainer = ModelTrainer(self.config.training)
            self.model = trainer.train(
                X_train_processed,
                y_train,
                self.config.data.protected_attributes
            )
            
            y_pred = trainer.predict(X_test_processed)
            
            validator = FinalValidator(
                self.config.fairness,
                self.config.data.protected_attributes
            )
            self.final_metrics = validator.validate(X_test, y_test, y_pred)
            validator.generate_comparison(self.baseline_metrics, self.final_metrics)
            
            mlflow_logger = MLflowLogger(self.config.mlflow)
            mlflow_logger.log_experiment(
                self.config_path,
                self.baseline_metrics,
                self.final_metrics,
                self.model,
                self.config.training.method,
                self.config.preprocessing.transformers,
                self.config.fairness.primary_metric
            )
            
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            logger.info("=" * 80 + "\n")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise


def main() -> None:
    """Main entry point for the pipeline orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fairness Pipeline Development Toolkit - Main Orchestrator'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yml',
        help='Path to configuration file (default: config.yml)'
    )
    
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator(config_path=args.config)
    orchestrator.run()


if __name__ == "__main__":
    main()

