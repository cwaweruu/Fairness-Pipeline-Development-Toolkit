# Quick Reference Guide

## Installation

```bash
git clone https://github.com/fairml-consulting/fairness-pipeline-toolkit.git
cd fairness-pipeline-toolkit
poetry install
poetry shell
```

## Basic Usage

```bash
python src/run_pipeline.py --config config.yml
```

## Configuration Template

```yaml
data:
  path: "data/your_dataset.csv"
  target_column: "target"
  protected_attributes: ["gender", "race"]
  test_size: 0.3
  random_state: 42

preprocessing:
  enabled: true
  transformers: ["DisparateImpactRemover"]
  repair_level: 0.8

training:
  method: "ReductionsWrapper"
  constraint: "DemographicParity"
  n_estimators: 100
  max_depth: 3
  learning_rate: 0.1

fairness:
  primary_metric: "demographic_parity_difference"
  threshold: 0.10

mlflow:
  experiment_name: "my_experiment"
  run_name_prefix: "run"
```

## Running Tests

```bash
poetry run pytest tests/
```

## Viewing MLflow UI

```bash
mlflow ui --port 5000
```

## Key Commands

| Command | Description |
|---------|-------------|
| `poetry install` | Install dependencies |
| `poetry shell` | Activate virtual environment |
| `python src/run_pipeline.py` | Run pipeline |
| `pytest tests/` | Run tests |
| `mlflow ui` | Start MLflow UI |
| `python scripts/generate_dataset.py` | Generate sample data |

## File Structure

```
fairness-pipeline-toolkit/
├── src/              # Source code
├── tests/            # Unit tests
├── data/             # Datasets
├── examples/         # Example notebooks
├── scripts/          # Utility scripts
├── config.yml        # Configuration
└── README.md         # Documentation
```

## Troubleshooting

**Import errors**: Ensure you're in virtual environment
**Config errors**: Check Pydantic validation messages
**MLflow issues**: Start server with `mlflow ui`

For more details, see README.md and USAGE.md
