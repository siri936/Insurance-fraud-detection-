"""
Database Manager Module
Handles MongoDB connections and operations for the fraud detection system
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional


class DatabaseManager:
    """Manages database operations for the fraud detection system"""
    
    def __init__(self, mongodb_uri: str):
        """
        Initialize database manager
        
        Args:
            mongodb_uri: MongoDB connection URI
        """
        self.uri = mongodb_uri
        self.connected = False
        self.claims_data = []
        self.predictions = []
        self.model_metadata = []
        
        # Try to connect (simplified - no actual MongoDB for now)
        try:
            self._validate_connection()
            self.connected = True
        except Exception as e:
            self.connected = False
            print(f"Warning: Could not connect to MongoDB: {e}")
    
    def _validate_connection(self):
        """Validate database connection"""
        # Placeholder for actual MongoDB connection
        if not self.uri:
            raise ValueError("MongoDB URI is required")
    
    def get_fraud_statistics(self) -> Dict[str, Any]:
        """
        Get fraud statistics from database
        
        Returns:
            Dictionary containing fraud statistics
        """
        fraud_count = sum(1 for c in self.claims_data if c.get('is_fraud', False))
        non_fraud_count = len(self.claims_data) - fraud_count
        total_predictions = len(self.predictions)
        
        # Calculate average risk score from predictions
        avg_risk_score = 0.0
        if self.predictions:
            avg_risk_score = sum(p.get('probability', 0) for p in self.predictions) / len(self.predictions)
        
        return {
            'total_claims': len(self.claims_data),
            'fraudulent_claims': fraud_count,
            'fraud_rate': fraud_count / len(self.claims_data) if len(self.claims_data) > 0 else 0,
            'avg_claim_amount': 5000.0,
            'total_predictions': total_predictions,
            'fraud_count': fraud_count,
            'non_fraud_count': non_fraud_count,
            'avg_risk_score': avg_risk_score
        }
    
    def get_all_claims(self) -> pd.DataFrame:
        """
        Retrieve all claims from database
        
        Returns:
            DataFrame with all claims
        """
        if self.claims_data:
            return pd.DataFrame(self.claims_data)
        else:
            # Return empty dataframe with expected columns
            return pd.DataFrame(columns=[
                'claim_id', 'claim_amount', 'claim_date', 'claimant_age',
                'policy_duration', 'is_fraud'
            ])
    
    def insert_claim(self, claim_data: Dict[str, Any]) -> bool:
        """
        Insert a claim into the database
        
        Args:
            claim_data: Dictionary containing claim information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            claim_data['created_at'] = datetime.now()
            self.claims_data.append(claim_data)
            return True
        except Exception as e:
            print(f"Error inserting claim: {e}")
            return False
    
    def insert_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """
        Insert a prediction record into the database
        
        Args:
            prediction_data: Dictionary containing prediction information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            prediction_data['created_at'] = datetime.now()
            self.predictions.append(prediction_data)
            return True
        except Exception as e:
            print(f"Error inserting prediction: {e}")
            return False
    
    def save_model_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Save model metadata to database
        
        Args:
            metadata: Dictionary containing model metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            metadata['saved_at'] = datetime.now()
            self.model_metadata.append(metadata)
            return True
        except Exception as e:
            print(f"Error saving model metadata: {e}")
            return False
    
    def get_model_metadata(self) -> List[Dict[str, Any]]:
        """
        Retrieve all model metadata
        
        Returns:
            List of model metadata dictionaries
        """
        return self.model_metadata
    
    def get_predictions_history(self, limit: int = 100) -> pd.DataFrame:
        """
        Get prediction history
        
        Args:
            limit: Maximum number of predictions to retrieve
            
        Returns:
            DataFrame with prediction history
        """
        if self.predictions:
            return pd.DataFrame(self.predictions[-limit:])
        else:
            return pd.DataFrame(columns=['claim_id', 'prediction', 'probability', 'created_at'])
    
    def get_all_predictions(self) -> pd.DataFrame:
        """
        Get all predictions made
        
        Returns:
            DataFrame with all predictions
        """
        if self.predictions:
            df = pd.DataFrame(self.predictions)
            # Rename probability to fraud_probability if needed
            if 'probability' in df.columns and 'fraud_probability' not in df.columns:
                df['fraud_probability'] = df['probability']
            # Ensure prediction column exists
            if 'prediction' not in df.columns:
                df['prediction'] = 0
            # Rename created_at to prediction_date if needed
            if 'created_at' in df.columns and 'prediction_date' not in df.columns:
                df['prediction_date'] = df['created_at']
            return df
        else:
            return pd.DataFrame(columns=['claim_id', 'prediction', 'fraud_probability', 'prediction_date'])
