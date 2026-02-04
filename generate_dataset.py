"""
Generate a sample loan approval dataset for demonstration purposes.

This script creates a realistic loan approval dataset that exhibits bias
patterns similar to those found in real-world lending data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_loan_approval_dataset(n_samples: int = 2000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic loan approval dataset with realistic bias patterns.
    
    Args:
        n_samples: Number of samples to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with loan approval data
    """
    np.random.seed(random_seed)
    
    data = {}
    
    data['gender'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45])
    data['race'] = np.random.choice(
        ['White', 'Black', 'Hispanic', 'Asian'],
        n_samples,
        p=[0.60, 0.15, 0.15, 0.10]
    )
    data['age'] = np.random.randint(22, 70, n_samples)
    data['age_group'] = pd.cut(
        data['age'],
        bins=[0, 30, 50, 100],
        labels=['18-30', '31-50', '50+']
    )
    
    data['income'] = np.random.lognormal(mean=11.0, sigma=0.5, size=n_samples)
    data['income'] = np.clip(data['income'], 20000, 300000)
    
    data['loan_amount'] = data['income'] * np.random.uniform(0.5, 3.0, n_samples)
    data['loan_amount'] = np.clip(data['loan_amount'], 5000, 500000)
    
    base_score = 300 + (data['income'] / 1000) * 2
    noise = np.random.normal(0, 50, n_samples)
    data['credit_score'] = np.clip(base_score + noise, 300, 850).astype(int)
    
    data['employment_years'] = np.random.exponential(scale=5, size=n_samples)
    data['employment_years'] = np.clip(data['employment_years'], 0, 40).astype(int)
    
    data['debt_to_income'] = data['loan_amount'] / data['income']
    
    approval_score = (
        (data['credit_score'] - 300) / 550 * 40 +
        (data['income'] / 300000) * 30 +
        (1 - data['debt_to_income'] / 5) * 20 +
        (data['employment_years'] / 40) * 10
    )
    
    bias_adjustments = np.zeros(n_samples)
    bias_adjustments[data['gender'] == 'Female'] -= 3
    bias_adjustments[data['race'] == 'Black'] -= 5
    bias_adjustments[data['race'] == 'Hispanic'] -= 3
    bias_adjustments[data['age_group'] == '18-30'] -= 2
    
    final_score = approval_score + bias_adjustments
    
    approval_threshold = 50
    noise = np.random.normal(0, 5, n_samples)
    data['loan_approved'] = ((final_score + noise) > approval_threshold).astype(int)
    
    data['years_at_current_address'] = np.random.randint(0, 20, n_samples)
    data['number_of_dependents'] = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.3, 0.3, 0.2, 0.15, 0.05])
    
    df = pd.DataFrame(data)
    
    df = df[[
        'loan_amount', 'income', 'credit_score', 'employment_years',
        'debt_to_income', 'years_at_current_address', 'number_of_dependents',
        'age', 'gender', 'race', 'age_group', 'loan_approved'
    ]]
    
    return df


def main():
    """Generate and save the dataset."""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    df = generate_loan_approval_dataset(n_samples=2000)
    
    output_path = data_dir / 'loan_approval.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated successfully: {output_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn types:")
    print(df.dtypes)
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nTarget distribution:")
    print(df['loan_approved'].value_counts(normalize=True))
    print(f"\nProtected attribute distributions:")
    print(f"Gender:\n{df['gender'].value_counts(normalize=True)}")
    print(f"\nRace:\n{df['race'].value_counts(normalize=True)}")
    print(f"\nAge Group:\n{df['age_group'].value_counts(normalize=True)}")


if __name__ == "__main__":
    main()
