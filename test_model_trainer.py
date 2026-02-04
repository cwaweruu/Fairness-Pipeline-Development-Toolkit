import pytest
import pandas as pd
import numpy as np
from src.model_trainer import ModelTrainer
from src.config_models import TrainingConfig


class TestModelTrainer:
    """Unit tests for ModelTrainer module."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        
        X_train = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'race': np.random.choice(['A', 'B'], n_samples)
        })
        
        y_train = pd.Series(np.random.choice([0, 1], n_samples))
        
        return X_train, y_train
    
    @pytest.fixture
    def valid_config(self):
        """Create a valid TrainingConfig."""
        return TrainingConfig(
            method="ReductionsWrapper",
            constraint="DemographicParity",
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            lambda_fairness=1.0
        )
    
    def test_initialization(self, valid_config):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(valid_config)
        assert trainer.config == valid_config
        assert trainer.model is None
        assert len(trainer.label_encoders) == 0
    
    def test_train(self, valid_config, sample_data):
        """Test model training."""
        X_train, y_train = sample_data
        trainer = ModelTrainer(valid_config)
        
        model = trainer.train(X_train, y_train, ['gender', 'race'])
        
        assert model is not None
        assert trainer.model is not None
        assert len(trainer.label_encoders) > 0
    
    def test_predict(self, valid_config, sample_data):
        """Test model prediction."""
        X_train, y_train = sample_data
        trainer = ModelTrainer(valid_config)
        
        trainer.train(X_train, y_train, ['gender', 'race'])
        predictions = trainer.predict(X_train)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_train)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_before_training(self, valid_config, sample_data):
        """Test that predict raises error if called before training."""
        X_train, y_train = sample_data
        trainer = ModelTrainer(valid_config)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            trainer.predict(X_train)
    
    def test_encoding_consistency(self, valid_config, sample_data):
        """Test that encoding is consistent between train and predict."""
        X_train, y_train = sample_data
        trainer = ModelTrainer(valid_config)
        
        trainer.train(X_train, y_train, ['gender', 'race'])
        
        X_test = X_train.copy()
        predictions = trainer.predict(X_test)
        
        assert len(predictions) == len(X_test)
    
    def test_config_validation(self):
        """Test that invalid configurations are rejected."""
        with pytest.raises(ValueError):
            TrainingConfig(
                method="InvalidMethod",
                constraint="DemographicParity",
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1
            )
        
        with pytest.raises(ValueError):
            TrainingConfig(
                method="ReductionsWrapper",
                constraint="InvalidConstraint",
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1
            )
