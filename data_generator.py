"""
Data Generator Module
Generates synthetic insurance claim data for model training and testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any


class FraudDataGenerator:
    """Generates realistic insurance claim data with fraud labels"""
    
    def __init__(self, n_samples: int = 1000, fraud_ratio: float = 0.15, random_state: int = 42):
        """
        Initialize data generator
        
        Args:
            n_samples: Number of samples to generate
            fraud_ratio: Ratio of fraudulent claims (0-1)
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.fraud_ratio = fraud_ratio
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate(self) -> pd.DataFrame:
        """
        Generate synthetic insurance claim data
        
        Returns:
            DataFrame with synthetic claims data
        """
        n_fraud = int(self.n_samples * self.fraud_ratio)
        n_legitimate = self.n_samples - n_fraud
        
        data = []
        
        # Generate legitimate claims
        for i in range(n_legitimate):
            claim = self._generate_legitimate_claim(i)
            claim['is_fraud'] = 0
            data.append(claim)
        
        # Generate fraudulent claims
        for i in range(n_fraud):
            claim = self._generate_fraud_claim(n_legitimate + i)
            claim['is_fraud'] = 1
            data.append(claim)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        return df
    
    def generate_data(self) -> pd.DataFrame:
        """
        Generate synthetic insurance claim data (alias for generate)
        
        Returns:
            DataFrame with synthetic claims data
        """
        return self.generate()
    
    def _generate_legitimate_claim(self, claim_id: int) -> Dict[str, Any]:
        """Generate a legitimate claim"""
        return {
            'claim_id': f'CLM_{claim_id:06d}',
            'claim_amount': np.random.lognormal(9, 1),  # Log-normal distribution
            'claim_date': datetime.now() - timedelta(days=np.random.randint(0, 365)),
            'claimant_age': np.random.normal(45, 15),
            'policy_duration': np.random.gamma(2, 2),
            'num_claims_per_year': np.random.poisson(1),
            'claim_review_time': np.random.lognormal(4.5, 1),
            'witness_count': np.random.poisson(2),
            'police_report': np.random.choice([0, 1], p=[0.3, 0.7]),
            'medical_assessment': np.random.choice([0, 1], p=[0.2, 0.8]),
        }
    
    def _generate_fraud_claim(self, claim_id: int) -> Dict[str, Any]:
        """Generate a fraudulent claim"""
        return {
            'claim_id': f'CLM_{claim_id:06d}',
            'claim_amount': np.random.lognormal(10, 1.5),  # Higher amounts
            'claim_date': datetime.now() - timedelta(days=np.random.randint(0, 365)),
            'claimant_age': np.random.normal(35, 10),  # Slightly younger average
            'policy_duration': np.random.gamma(1, 1),  # Shorter policy duration
            'num_claims_per_year': np.random.poisson(3),  # More frequent claims
            'claim_review_time': np.random.lognormal(3.5, 0.8),  # Faster review
            'witness_count': np.random.poisson(0.5),  # Fewer witnesses
            'police_report': np.random.choice([0, 1], p=[0.8, 0.2]),  # Less likely
            'medical_assessment': np.random.choice([0, 1], p=[0.7, 0.3]),  # Less likely
        }
    
    def generate_with_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate data and split into features and targets
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = self.generate()
        
        feature_cols = [col for col in df.columns if col not in ['claim_id', 'is_fraud', 'claim_date']]
        X = df[feature_cols].astype(float)
        y = df['is_fraud']
        
        return df, y
