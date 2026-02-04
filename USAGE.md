# Usage Guide

This guide provides detailed instructions for using the Fairness Pipeline Development Toolkit.

## Table of Contents

1. [Configuration Guide](#configuration-guide)
2. [Running the Pipeline](#running-the-pipeline)
3. [Interpreting Results](#interpreting-results)
4. [MLflow Integration](#mlflow-integration)
5. [Customization](#customization)
6. [Advanced Topics](#advanced-topics)

## Configuration Guide

The `config.yml` file controls all aspects of the pipeline. Here's a detailed breakdown:

### Data Configuration

```yaml
data:
  path: "data/loan_approval.csv"     # Path to your dataset
  target_column: "loan_approved"      # Column to predict
  protected_attributes:               # Sensitive attributes to monitor
    - "gender"
    - "race"
    - "age_group"
  test_size: 0.3                      # Proportion for test set (0.0-1.0)
  random_state: 42                    # Random seed for reproducibility
```

**Required Fields:**
- `path`: Path to CSV dataset
- `target_column`: Name of target variable (must exist in dataset)
- `protected_attributes`: At least one protected attribute

**Optional Fields:**
- `test_size`: Default 0.3 (30% test data)
- `random_state`: Default 42

### Preprocessing Configuration

```yaml
preprocessing:
  enabled: true                       # Enable/disable preprocessing
  transformers:                       # List of transformers to apply
    - "DisparateImpactRemover"
  repair_level: 0.8                   # Bias mitigation strength (0.0-1.0)
```

**Available Transformers:**
- `DisparateImpactRemover`: Reduces correlation with protected attributes
- Additional transformers can be added in `src/preprocessing.py`

**Repair Level:**
- `0.0`: No repair (baseline)
- `0.5`: Moderate repair
- `1.0`: Maximum repair (may impact performance)

### Training Configuration

```yaml
training:
  method: "ReductionsWrapper"         # Training approach
  constraint: "DemographicParity"     # Fairness constraint
  n_estimators: 100                   # Number of trees
  max_depth: 3                        # Maximum tree depth
  learning_rate: 0.1                  # Learning rate
  lambda_fairness: 1.0                # Fairness regularization strength
```

**Valid Methods:**
- `ReductionsWrapper`: Fairness-constrained optimization
- `FairnessRegularizer`: Regularization-based fairness
- `Baseline`: Standard training without constraints

**Valid Constraints:**
- `DemographicParity`: Equal selection rates across groups
- `EqualizedOdds`: Equal true/false positive rates
- `EqualOpportunity`: Equal true positive rates only

### Fairness Configuration

```yaml
fairness:
  primary_metric: "demographic_parity_difference"  # Main metric to optimize
  threshold: 0.10                                   # Acceptable threshold
  secondary_metrics:                                # Additional metrics to track
    - "equal_opportunity_difference"
    - "average_odds_difference"
```

**Valid Metrics:**
- `demographic_parity_difference`: Difference in selection rates
- `equalized_odds_difference`: Difference in TPR and FPR
- `equal_opportunity_difference`: Difference in TPR only
- `disparate_impact_ratio`: Ratio of selection rates

**Threshold Interpretation:**
- For `*_difference` metrics: closer to 0 is better, typically ≤ 0.10
- For `disparate_impact_ratio`: closer to 1.0 is better, typically ≥ 0.80

### MLflow Configuration

```yaml
mlflow:
  experiment_name: "fairness_pipeline_toolkit"
  run_name_prefix: "pipeline_run"
  tags:
    project: "Fairness Pipeline Development"
    team: "FairML Consulting"
    version: "1.0.0"
    domain: "finance"
```

## Running the Pipeline

### Basic Usage

```bash
python src/run_pipeline.py --config config.yml
```

### Custom Configuration

```bash
python src/run_pipeline.py --config custom_config.yml
```

### From Python

```python
from src.run_pipeline import PipelineOrchestrator

orchestrator = PipelineOrchestrator(config_path="config.yml")
orchestrator.run()

print(f"Baseline metrics: {orchestrator.baseline_metrics}")
print(f"Final metrics: {orchestrator.final_metrics}")
```

## Interpreting Results

### Console Output

The pipeline logs detailed progress to both console and `pipeline_run.log`:

1. **Configuration Loading**: Validates config.yml
2. **Data Loading**: Reports dataset shape and split sizes
3. **Baseline Measurement**: Shows initial fairness metrics
4. **Preprocessing**: Confirms transformations applied
5. **Model Training**: Reports training progress
6. **Final Validation**: Displays performance and fairness metrics
7. **Comparison Report**: Shows before/after improvement
8. **MLflow Logging**: Confirms artifact storage

### Understanding Metrics

**Performance Metrics:**
- `accuracy`: Overall prediction accuracy
- `precision`: Positive predictive value
- `recall`: True positive rate
- `f1_score`: Harmonic mean of precision and recall

**Fairness Metrics:**
- `demographic_parity_difference`: |selection_rate_group1 - selection_rate_group2|
- Lower values indicate better fairness
- Threshold typically set at 0.05-0.10

### Success Criteria

Pipeline succeeds when:
1. Final fairness metric ≤ threshold
2. Performance metrics meet minimum requirements
3. Improvement over baseline is achieved

## MLflow Integration

### Starting MLflow UI

```bash
mlflow ui --port 5000
```

Then navigate to http://localhost:5000

### Viewing Experiments

In the MLflow UI:
1. Select your experiment from the left sidebar
2. View all runs in chronological order
3. Compare metrics across runs
4. Download artifacts (models, configs)

### Querying Programmatically

```python
import mlflow

experiment = mlflow.get_experiment_by_name("fairness_pipeline_toolkit")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.final_accuracy > 0.8"
)
print(runs)
```

### Comparing Runs

```python
run_ids = ["run1_id", "run2_id"]

for run_id in run_ids:
    run = mlflow.get_run(run_id)
    print(f"Run {run_id}:")
    print(f"  Accuracy: {run.data.metrics['final_accuracy']}")
    print(f"  Fairness: {run.data.metrics['final_demographic_parity_difference']}")
```

## Customization

### Adding Custom Preprocessing

Edit `src/preprocessing.py`:

```python
class CustomTransformer:
    def __init__(self, param1: float):
        self.param1 = param1
    
    def fit_transform(self, X, y=None):
        # Your transformation logic
        return X_transformed
```

Update config.yml:
```yaml
preprocessing:
  transformers:
    - "CustomTransformer"
```

### Adding Custom Fairness Metrics

Edit `src/final_validator.py`:

```python
def custom_fairness_metric(y_true, y_pred, sensitive_attr):
    # Your metric calculation
    return metric_value
```

Update config.yml:
```yaml
fairness:
  primary_metric: "custom_fairness_metric"
```

### Using Different Models

Edit `src/model_trainer.py` to use different base estimators:

```python
from sklearn.ensemble import RandomForestClassifier

self.model = RandomForestClassifier(
    n_estimators=self.config.n_estimators,
    max_depth=self.config.max_depth
)
```

## Advanced Topics

### Batch Experimentation

```python
configs = [
    "config_baseline.yml",
    "config_medium.yml",
    "config_aggressive.yml"
]

results = []
for config_path in configs:
    orchestrator = PipelineOrchestrator(config_path=config_path)
    orchestrator.run()
    results.append({
        'config': config_path,
        'fairness': orchestrator.final_metrics['demographic_parity_difference'],
        'accuracy': orchestrator.final_metrics['accuracy']
    })

import pandas as pd
df_results = pd.DataFrame(results)
print(df_results)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 3, 5],
    'learning_rate': [0.05, 0.1, 0.2]
}

# Implement in model_trainer.py
```

### Integration with CI/CD

```bash
#!/bin/bash
# ci_fairness_check.sh

python src/run_pipeline.py --config config.yml

if [ $? -eq 0 ]; then
    echo "Fairness pipeline passed"
    exit 0
else
    echo "Fairness pipeline failed"
    exit 1
fi
```

### Multi-Attribute Fairness

Monitor fairness across multiple protected attributes:

```yaml
data:
  protected_attributes:
    - "gender"
    - "race"
    - "age_group"
```

The pipeline will analyze each attribute separately.

## Troubleshooting

### Common Issues

**Issue: Dataset not found**
```
FileNotFoundError: Data file not found: data/dataset.csv
```
Solution: Verify the path in config.yml matches your data location

**Issue: Protected attribute missing**
```
KeyError: 'gender'
```
Solution: Ensure all protected attributes specified in config exist in dataset

**Issue: Validation errors**
```
ValidationError: method must be one of ['ReductionsWrapper', ...]
```
Solution: Check config.yml for typos in field values

### Getting Help

1. Check the logs in `pipeline_run.log`
2. Review error messages carefully
3. Verify configuration against examples
4. Open an issue on GitHub with error details

## Best Practices

1. **Start Simple**: Begin with baseline configuration before optimization
2. **Version Control**: Track all config files in git
3. **Document Changes**: Comment why specific parameters were chosen
4. **Validate Regularly**: Re-run fairness checks periodically
5. **Monitor Production**: Continue tracking fairness after deployment

