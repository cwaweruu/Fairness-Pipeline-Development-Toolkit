import pytest
import pandas as pd
import numpy as np
from src.preprocessing import PreprocessingPipeline
from src.config_models import PreprocessingConfig


class TestPreprocessingPipeline:
    """Unit tests for PreprocessingPipeline module."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training and test data."""
        np.random.seed(42)
        n_train, n_test = 100, 30
        
        X_train = pd.DataFrame({
            'feature1': np.random.randn(n_train) * 10 + 50,
            'feature2': np.random.randn(n_train) * 5 + 20,
            'gender': np.random.choice(['M', 'F'], n_train),
            'race': np.random.choice(['A', 'B'], n_train)
        })
        
        X_test = pd.DataFrame({
            'feature1': np.random.randn(n_test) * 10 + 50,
            'feature2': np.random.randn(n_test) * 5 + 20,
            'gender': np.random.choice(['M', 'F'], n_test),
            'race': np.random.choice(['A', 'B'], n_test)
        })
        
        y_train = pd.Series(np.random.choice([0, 1], n_train))
        
        return X_train, X_test, y_train
    
    @pytest.fixture
    def enabled_config(self):
        """Create preprocessing config with preprocessing enabled."""
        return PreprocessingConfig(
            enabled=True,
            transformers=["DisparateImpactRemover"],
            repair_level=0.8
        )
    
    @pytest.fixture
    def disabled_config(self):
        """Create preprocessing config with preprocessing disabled."""
        return PreprocessingConfig(
            enabled=False,
            transformers=[],
            repair_level=0.0
        )
    
    def test_initialization(self, enabled_config):
        """Test PreprocessingPipeline initialization."""
        pipeline = PreprocessingPipeline(enabled_config)
        assert pipeline.config == enabled_config
        assert pipeline.scaler is not None
    
    def test_transform_enabled(self, enabled_config, sample_data):
        """Test transformation when preprocessing is enabled."""
        X_train, X_test, y_train = sample_data
        pipeline = PreprocessingPipeline(enabled_config)
        
        X_train_proc, X_test_proc = pipeline.transform(
            X_train, X_test, y_train, ['gender', 'race']
        )
        
        assert isinstance(X_train_proc, pd.DataFrame)
        assert isinstance(X_test_proc, pd.DataFrame)
        assert X_train_proc.shape == X_train.shape
        assert X_test_proc.shape == X_test.shape
    
    def test_transform_disabled(self, disabled_config, sample_data):
        """Test that transformation is skipped when disabled."""
        X_train, X_test, y_train = sample_data
        pipeline = PreprocessingPipeline(disabled_config)
        
        X_train_proc, X_test_proc = pipeline.transform(
            X_train, X_test, y_train, ['gender', 'race']
        )
        
        pd.testing.assert_frame_equal(X_train_proc, X_train)
        pd.testing.assert_frame_equal(X_test_proc, X_test)
    
    def test_scaling_applied(self, enabled_config, sample_data):
        """Test that numeric features are scaled."""
        X_train, X_test, y_train = sample_data
        pipeline = PreprocessingPipeline(enabled_config)
        
        X_train_proc, X_test_proc = pipeline.transform(
            X_train, X_test, y_train, ['gender', 'race']
        )
        
        assert abs(X_train_proc['feature1'].mean()) < 0.1
        assert abs(X_train_proc['feature1'].std() - 1.0) < 0.1
    
    def test_categorical_preserved(self, enabled_config, sample_data):
        """Test that categorical features are preserved."""
        X_train, X_test, y_train = sample_data
        pipeline = PreprocessingPipeline(enabled_config)
        
        X_train_proc, X_test_proc = pipeline.transform(
            X_train, X_test, y_train, ['gender', 'race']
        )
        
        assert set(X_train_proc['gender'].unique()) == set(X_train['gender'].unique())
        assert set(X_train_proc['race'].unique()) == set(X_train['race'].unique())
