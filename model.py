"""
Fraud Detection Model Module
Implements machine learning models for insurance fraud detection
"""

import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from typing import Dict, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionModel:
    """Machine learning models for fraud detection"""
    
    def __init__(self):
        """Initialize the fraud detection model"""
        self.scaler = StandardScaler()
        self.models = {}
        self.best_model_name = None
        self.best_model = None
        self.X_test = None
        self.y_test = None
        self.feature_columns = None
        self.feature_importance_data = None
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess data for model training
        
        Args:
            data: DataFrame with claim data
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Handle missing values if any
        data = data.fillna(data.mean(numeric_only=True))
        
        # Separate features and target
        if 'is_fraud' in data.columns:
            y = data['is_fraud']
            X = data.drop(columns=['is_fraud', 'claim_id', 'claim_date'], errors='ignore')
        else:
            X = data.drop(columns=['claim_id', 'claim_date'], errors='ignore')
            y = None
        
        # Convert to float
        X = X.astype(float)
        
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, use_smote: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Train multiple machine learning models
        
        Args:
            X: Features DataFrame
            y: Target Series
            use_smote: Whether to use SMOTE for class balancing
            
        Returns:
            Dictionary with results for each model
        """
        # Handle class imbalance
        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42)
                X, y = smote.fit_resample(X, y)
            except ImportError:
                pass
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # Define models to train
        models_to_train = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        results = {}
        
        for model_name, model in models_to_train.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            self.models[model_name] = model
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            results[model_name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'fpr': None,
                'tpr': None,
                'y_pred_proba': y_pred_proba
            }
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            results[model_name]['fpr'] = fpr
            results[model_name]['tpr'] = tpr
        
        # Select best model based on ROC-AUC
        self.best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
        self.best_model = self.models[self.best_model_name]
        
        # Store feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            self.feature_importance_data = importance_df
        
        return results
    
    def predict(self, input_data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make a fraud prediction for a single claim
        
        Args:
            input_data: Dictionary or DataFrame with claim features
            
        Returns:
            Tuple of (predictions array, probabilities array)
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet")
        
        # Convert DataFrame to dictionary if needed
        if isinstance(input_data, pd.DataFrame):
            input_dict = input_data.to_dict('records')[0]
        else:
            input_dict = input_data
        
        # Create feature vector with proper type conversion
        # Only use features that were used in training
        feature_vector = []
        
        if self.feature_columns is None or len(self.feature_columns) == 0:
            # Fallback: handle default feature names from data_generator
            default_features = [
                'claim_amount', 'claimant_age', 'policy_duration', 
                'num_claims_per_year', 'claim_review_time', 'witness_count',
                'police_report', 'medical_assessment'
            ]
            feature_cols_to_use = default_features
        else:
            feature_cols_to_use = self.feature_columns
        
        for col in feature_cols_to_use:
            value = input_dict.get(col, 0)
            
            # Map prediction page fields to training features
            if col == 'claim_amount' and 'claim_amount' not in input_dict:
                value = input_dict.get('claim_amount', 5000)
            elif col == 'claimant_age' and 'claimant_age' not in input_dict:
                value = input_dict.get('age', 35)
            elif col == 'policy_duration' and 'policy_duration' not in input_dict:
                value = input_dict.get('policy_tenure', 5)
            elif col == 'witness_count' and 'witness_count' not in input_dict:
                value = input_dict.get('witnesses', 1)
            elif col == 'police_report' and 'police_report' not in input_dict:
                value = input_dict.get('police_report_filed', 0)
            
            # Convert categorical values to numeric if needed
            if isinstance(value, str):
                # Map common values
                if value.lower() in ['yes', 'true', '1']:
                    feature_vector.append(1.0)
                elif value.lower() in ['no', 'false', '0']:
                    feature_vector.append(0.0)
                elif value.lower() == 'minor':
                    feature_vector.append(1.0)
                elif value.lower() == 'moderate':
                    feature_vector.append(2.0)
                elif value.lower() == 'major':
                    feature_vector.append(3.0)
                else:
                    feature_vector.append(0.0)
            else:
                # Ensure numeric value
                try:
                    feature_vector.append(float(value))
                except (ValueError, TypeError):
                    feature_vector.append(0.0)
        
        # Create numpy array with proper dtype
        X = np.array([feature_vector], dtype=np.float64)
        
        # Scale features
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            # If scaler fails, just use raw data
            X_scaled = X
        
        # Make prediction
        predictions = self.best_model.predict(X_scaled)
        probabilities = self.best_model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def save_model(self, filename: str) -> bool:
        """
        Save the trained model to disk
        
        Args:
            filename: Path to save the model
            
        Returns:
            True if successful
        """
        try:
            model_data = {
                'best_model': self.best_model,
                'best_model_name': self.best_model_name,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filename: str) -> bool:
        """
        Load a trained model from disk
        
        Args:
            filename: Path to load the model from
            
        Returns:
            True if successful
        """
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            self.best_model = model_data['best_model']
            self.best_model_name = model_data['best_model_name']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the best model
        
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance_data is None:
            return pd.DataFrame({'feature': self.feature_columns, 'importance': [0] * len(self.feature_columns)})
        return self.feature_importance_data
    
    def generate_model_comparison_plot(self, results: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        Generate a model comparison plot
        
        Args:
            results: Dictionary with model results
            
        Returns:
            Plotly figure
        """
        model_names = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [results[model][metric] for model in model_names]
            fig.add_trace(go.Bar(x=model_names, y=values, name=metric.capitalize()))
        
        fig.update_layout(
            title='Model Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=500
        )
        
        return fig
    
    def generate_confusion_matrix_plot(self, model_name: str, results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """
        Generate a confusion matrix heatmap
        
        Args:
            model_name: Name of the model
            results: Dictionary with model results
            
        Returns:
            Plotly figure
        """
        cm = results[model_name]['confusion_matrix']
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Legitimate', 'Fraudulent'],
            y=['Legitimate', 'Fraudulent'],
            text=cm,
            texttemplate="%{text}",
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=500
        )
        
        return fig
    
    def generate_roc_curve_plot(self, results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """
        Generate ROC curves for all models
        
        Args:
            results: Dictionary with model results
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for model_name, model_results in results.items():
            fpr = model_results['fpr']
            tpr = model_results['tpr']
            roc_auc = model_results['roc_auc']
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC={roc_auc:.4f})'
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curves',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500
        )
        
        return fig
