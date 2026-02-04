from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for data loading and preparation."""
    
    path: Optional[str] = Field(None, description="Path to dataset CSV file")
    target_column: str = Field(..., description="Name of target variable")
    protected_attributes: List[str] = Field(..., description="List of protected attributes")
    test_size: float = Field(0.3, ge=0.0, le=1.0, description="Test set proportion")
    random_state: int = Field(42, description="Random seed for reproducibility")
    
    @field_validator('test_size')
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError('test_size must be between 0 and 1')
        return v


class PreprocessingConfig(BaseModel):
    """Configuration for preprocessing transformations."""
    
    enabled: bool = Field(True, description="Enable/disable preprocessing")
    transformers: List[str] = Field(
        default_factory=list,
        description="List of transformer names to apply"
    )
    repair_level: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Repair level for disparate impact remover"
    )


class TrainingConfig(BaseModel):
    """Configuration for model training."""
    
    method: str = Field(..., description="Training method to use")
    constraint: str = Field(..., description="Fairness constraint type")
    n_estimators: int = Field(100, gt=0, description="Number of estimators")
    max_depth: int = Field(3, gt=0, description="Maximum tree depth")
    learning_rate: float = Field(0.1, gt=0.0, description="Learning rate")
    lambda_fairness: float = Field(1.0, ge=0.0, description="Fairness regularization strength")
    
    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        valid_methods = ['ReductionsWrapper', 'FairnessRegularizer', 'Baseline']
        if v not in valid_methods:
            raise ValueError(f'method must be one of {valid_methods}')
        return v
    
    @field_validator('constraint')
    @classmethod
    def validate_constraint(cls, v: str) -> str:
        valid_constraints = ['DemographicParity', 'EqualizedOdds', 'EqualOpportunity']
        if v not in valid_constraints:
            raise ValueError(f'constraint must be one of {valid_constraints}')
        return v


class FairnessConfig(BaseModel):
    """Configuration for fairness metrics and validation."""
    
    primary_metric: str = Field(..., description="Primary fairness metric")
    threshold: float = Field(..., gt=0.0, description="Acceptable threshold")
    secondary_metrics: List[str] = Field(
        default_factory=list,
        description="Additional metrics to monitor"
    )
    
    @field_validator('primary_metric')
    @classmethod
    def validate_metric(cls, v: str) -> str:
        valid_metrics = [
            'demographic_parity_difference',
            'equalized_odds_difference',
            'equal_opportunity_difference',
            'disparate_impact_ratio'
        ]
        if v not in valid_metrics:
            raise ValueError(f'primary_metric must be one of {valid_metrics}')
        return v


class MLflowConfig(BaseModel):
    """Configuration for MLflow experiment tracking."""
    
    experiment_name: str = Field(
        "fairness_pipeline_toolkit",
        description="MLflow experiment name"
    )
    tracking_uri: Optional[str] = Field(None, description="MLflow tracking URI")
    run_name_prefix: str = Field("pipeline_run", description="Run name prefix")
    tags: Dict[str, str] = Field(default_factory=dict, description="Run tags")


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    
    data: DataConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    fairness: FairnessConfig
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    
    class Config:
        extra = "allow"
