import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import DataLoader
from src.config_models import DataConfig


class TestDataLoader:
    """Unit tests for DataLoader module."""
    
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a temporary CSV file for testing."""
        data = {
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'gender': np.random.choice(['M', 'F'], 100),
            'race': np.random.choice(['A', 'B'], 100),
            'target': np.random.choice([0, 1], 100)
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path
    
    @pytest.fixture
    def valid_config(self, sample_csv):
        """Create a valid DataConfig."""
        return DataConfig(
            path=str(sample_csv),
            target_column="target",
            protected_attributes=["gender", "race"],
            test_size=0.3,
            random_state=42
        )
    
    def test_initialization(self, valid_config):
        """Test DataLoader initialization."""
        loader = DataLoader(valid_config)
        assert loader.config == valid_config
    
    def test_load_and_split(self, valid_config):
        """Test data loading and splitting."""
        loader = DataLoader(valid_config)
        X_train, X_test, y_train, y_test = loader.load_and_split()
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        
        total_samples = len(X_train) + len(X_test)
        assert total_samples == 100
        
        expected_test_size = int(100 * 0.3)
        assert len(X_test) == expected_test_size
    
    def test_missing_file(self):
        """Test error handling for missing file."""
        config = DataConfig(
            path="nonexistent.csv",
            target_column="target",
            protected_attributes=["gender"]
        )
        loader = DataLoader(config)
        
        with pytest.raises(FileNotFoundError):
            loader.load_and_split()
    
    def test_invalid_target_column(self, sample_csv):
        """Test error handling for invalid target column."""
        config = DataConfig(
            path=str(sample_csv),
            target_column="nonexistent",
            protected_attributes=["gender"]
        )
        loader = DataLoader(config)
        
        with pytest.raises(ValueError):
            loader.load_and_split()
    
    def test_reproducibility(self, valid_config):
        """Test that same random_state produces same split."""
        loader1 = DataLoader(valid_config)
        X_train1, X_test1, y_train1, y_test1 = loader1.load_and_split()
        
        loader2 = DataLoader(valid_config)
        X_train2, X_test2, y_train2, y_test2 = loader2.load_and_split()
        
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_series_equal(y_train1, y_train2)
