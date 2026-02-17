"""
AI-Powered Insurance Fraud Detection System
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Import custom modules
from database import DatabaseManager
from data_generator import FraudDataGenerator
from model import FraudDetectionModel

# Page configuration
st.set_page_config(
    page_title="Insurance Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .safe-alert {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# MongoDB connection string
MONGODB_URI = "mongodb+srv://kallasrimanikantaram_db_user:SzBlUsNN3IfS1WgU@cluster0.gjpjx2d.mongodb.net/"

# Initialize session state
if 'db_manager' not in st.session_state:
    try:
        st.session_state.db_manager = DatabaseManager(MONGODB_URI)
        st.session_state.db_connected = True
    except Exception as e:
        st.session_state.db_connected = False
        st.error(f"Failed to connect to MongoDB: {e}")

if 'model' not in st.session_state:
    st.session_state.model = FraudDetectionModel()

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

if 'training_results' not in st.session_state:
    st.session_state.training_results = None


def main():
    """Main application function"""
    
    # Header
    st.markdown('<p class="main-header">🔍 AI-Powered Insurance Fraud Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detect fraudulent insurance claims using advanced machine learning</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "📊 Data Management", "🤖 Model Training", "🔮 Fraud Prediction", "📈 Analytics Dashboard", "ℹ️ About"]
    )
    
    # Database status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Database Status")
    if st.session_state.db_connected:
        st.sidebar.success("✅ Connected to MongoDB")
    else:
        st.sidebar.error("❌ Database Disconnected")
    
    # Model status
    st.sidebar.markdown("### Model Status")
    if st.session_state.model_trained:
        st.sidebar.success(f"✅ Model Trained ({st.session_state.model.best_model_name})")
    else:
        st.sidebar.warning("⚠️ Model Not Trained")
    
    # Route to selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Management":
        show_data_management_page()
    elif page == "🤖 Model Training":
        show_model_training_page()
    elif page == "🔮 Fraud Prediction":
        show_prediction_page()
    elif page == "📈 Analytics Dashboard":
        show_analytics_page()
    elif page == "ℹ️ About":
        show_about_page()


def show_home_page():
    """Display home page"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Objectives</h3>
            <ul>
                <li>Detect fraudulent claims</li>
                <li>Analyze claim patterns</li>
                <li>Risk scoring system</li>
                <li>Decision support</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Key Features</h3>
            <ul>
                <li>ML-based classification</li>
                <li>Pattern analysis</li>
                <li>Risk probability scoring</li>
                <li>Real-time predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🛠️ Technologies</h3>
            <ul>
                <li>Python & Streamlit</li>
                <li>Scikit-learn & XGBoost</li>
                <li>MongoDB Database</li>
                <li>Plotly Visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick stats
    if st.session_state.db_connected:
        stats = st.session_state.db_manager.get_fraud_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", stats['total_predictions'])
        
        with col2:
            st.metric("Fraud Detected", stats['fraud_count'])
        
        with col3:
            st.metric("Legitimate Claims", stats['non_fraud_count'])
        
        with col4:
            avg_score = stats['avg_risk_score']
            st.metric("Avg Risk Score", f"{avg_score:.2%}" if avg_score else "N/A")
    
    st.markdown("---")
    
    # Getting started guide
    st.markdown("### 🚀 Getting Started")
    
    st.markdown("""
    1. **Generate Data**: Go to Data Management to generate synthetic insurance claim data
    2. **Train Model**: Navigate to Model Training to train fraud detection models
    3. **Make Predictions**: Use the Fraud Prediction page to analyze new claims
    4. **View Analytics**: Check the Analytics Dashboard for insights and trends
    """)


def show_data_management_page():
    """Data management page"""
    
    st.header("📊 Data Management")
    
    tab1, tab2, tab3 = st.tabs(["Generate Data", "View Data", "Upload Data"])
    
    with tab1:
        st.subheader("Generate Synthetic Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_samples = st.number_input("Number of Samples", min_value=100, max_value=10000, value=2000, step=100)
        
        with col2:
            fraud_ratio = st.slider("Fraud Ratio", min_value=0.05, max_value=0.5, value=0.15, step=0.05)
        
        if st.button("🔄 Generate Data", type="primary"):
            with st.spinner("Generating synthetic data..."):
                generator = FraudDataGenerator(n_samples=n_samples, fraud_ratio=fraud_ratio)
                data = generator.generate_data()
                
                # Save to session state
                st.session_state.training_data = data
                
                # Save to CSV
                data.to_csv('insurance_claims_data.csv', index=False)
                
                st.success(f"✅ Generated {len(data)} samples!")
                
                # Show preview
                st.dataframe(data.head(10))
                
                # Show statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Samples", len(data))
                    st.metric("Fraud Cases", data['is_fraud'].sum())
                
                with col2:
                    st.metric("Legitimate Cases", len(data) - data['is_fraud'].sum())
                    st.metric("Fraud Percentage", f"{data['is_fraud'].mean()*100:.2f}%")
    
    with tab2:
        st.subheader("View Existing Data")
        
        if st.session_state.db_connected:
            claims_df = st.session_state.db_manager.get_all_claims()
            
            if not claims_df.empty:
                st.dataframe(claims_df)
                
                # Download button
                csv = claims_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv,
                    file_name="claims_data.csv",
                    mime="text/csv"
                )
            else:
                st.info("No claims data available in database.")
        else:
            st.error("Database not connected!")
    
    with tab3:
        st.subheader("Upload Training Data")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.session_state.training_data = data
            
            st.success("✅ Data uploaded successfully!")
            st.dataframe(data.head())
            
            # Show info
            st.info(f"Loaded {len(data)} records with {len(data.columns)} columns")


def show_model_training_page():
    """Model training page"""
    
    st.header("🤖 Model Training")
    
    # Check if data is available
    if 'training_data' not in st.session_state or st.session_state.training_data is None:
        st.warning("⚠️ No training data available. Please generate or upload data first.")
        
        if st.button("Go to Data Management"):
            st.session_state.page = "📊 Data Management"
        return
    
    data = st.session_state.training_data
    
    st.info(f"Training data loaded: {len(data)} samples")
    
    # Training options
    col1, col2 = st.columns(2)
    
    with col1:
        use_smote = st.checkbox("Use SMOTE for class balancing", value=True)
    
    with col2:
        st.metric("Fraud Ratio", f"{data['is_fraud'].mean()*100:.2f}%")
    
    # Train button
    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            
            # Preprocess data
            X, y = st.session_state.model.preprocess_data(data)
            
            # Train models
            results = st.session_state.model.train_models(X, y, use_smote=use_smote)
            
            # Store results
            st.session_state.training_results = results
            st.session_state.model_trained = True
            
            # Save model
            st.session_state.model.save_model('fraud_detection_model.pkl')
            
            # Save metadata to database
            if st.session_state.db_connected:
                model_metadata = {
                    'model_name': st.session_state.model.best_model_name,
                    'accuracy': results[st.session_state.model.best_model_name]['accuracy'],
                    'roc_auc': results[st.session_state.model.best_model_name]['roc_auc'],
                    'training_samples': len(data),
                    'fraud_ratio': data['is_fraud'].mean()
                }
                st.session_state.db_manager.save_model_metadata(model_metadata)
            
            st.success("✅ Models trained successfully!")
    
    # Display results if available
    if st.session_state.training_results is not None:
        st.markdown("---")
        st.subheader("📊 Model Performance Results")
        
        results = st.session_state.training_results
        
        # Create metrics comparison table
        metrics_data = []
        for model_name, metrics in results.items():
            metrics_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Best model highlight
        st.success(f"🏆 Best Model: **{st.session_state.model.best_model_name}** (ROC-AUC: {results[st.session_state.model.best_model_name]['roc_auc']:.4f})")
        
        # Visualizations
        st.markdown("---")
        st.subheader("📈 Model Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Performance Comparison", "Confusion Matrix", "ROC Curve", "Feature Importance"])
        
        with tab1:
            fig = st.session_state.model.generate_model_comparison_plot(results)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            model_choice = st.selectbox("Select Model", list(results.keys()))
            fig = st.session_state.model.generate_confusion_matrix_plot(model_choice, results)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            fig = st.session_state.model.generate_roc_curve_plot(results)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            importance_df = st.session_state.model.get_feature_importance()
            if not importance_df.empty:
                fig = px.bar(
                    importance_df.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 10 Most Important Features',
                    labels={'importance': 'Importance Score', 'feature': 'Feature'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")


def show_prediction_page():
    """Fraud prediction page"""
    
    st.header("🔮 Fraud Prediction")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first before making predictions.")
        return
    
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with tab1:
        st.subheader("Enter Claim Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            policy_tenure = st.number_input("Policy Tenure (years)", min_value=0, max_value=50, value=5)
            claim_amount = st.number_input("Claim Amount ($)", min_value=0, max_value=100000, value=5000)
            previous_claims = st.number_input("Previous Claims", min_value=0, max_value=20, value=1)
        
        with col2:
            incident_severity = st.selectbox("Incident Severity", ['Minor', 'Moderate', 'Major'])
            witnesses = st.number_input("Number of Witnesses", min_value=0, max_value=10, value=1)
            police_report = st.selectbox("Police Report Filed", ['No', 'Yes'])
            incident_type = st.selectbox("Incident Type", ['Collision', 'Theft', 'Fire', 'Vandalism'])
        
        with col3:
            vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=50, value=5)
            vehicles_involved = st.number_input("Vehicles Involved", min_value=1, max_value=5, value=1)
            bodily_injuries = st.number_input("Bodily Injuries", min_value=0, max_value=10, value=0)
            claim_hour = st.slider("Claim Hour (0-23)", min_value=0, max_value=23, value=12)
        
        claim_day = st.slider("Claim Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=3)
        
        if st.button("🔍 Predict Fraud Risk", type="primary"):
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'policy_tenure': [policy_tenure],
                'claim_amount': [claim_amount],
                'previous_claims': [previous_claims],
                'incident_severity': [incident_severity],
                'witnesses': [witnesses],
                'police_report_filed': [1 if police_report == 'Yes' else 0],
                'incident_type': [incident_type],
                'vehicle_age': [vehicle_age],
                'number_of_vehicles_involved': [vehicles_involved],
                'bodily_injuries': [bodily_injuries],
                'claim_hour': [claim_hour],
                'claim_day_of_week': [claim_day]
            })
            
            # Make prediction
            prediction, probability = st.session_state.model.predict(input_data)
            
            fraud_prob = probability[0]
            is_fraud = prediction[0]
            
            # Display result
            st.markdown("---")
            st.subheader("Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if is_fraud == 1:
                    st.markdown(f"""
                    <div class="fraud-alert">
                        <h3>⚠️ FRAUD ALERT</h3>
                        <p>This claim has been flagged as potentially <strong>FRAUDULENT</strong></p>
                        <p><strong>Fraud Probability: {fraud_prob:.2%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-alert">
                        <h3>✅ LEGITIMATE CLAIM</h3>
                        <p>This claim appears to be <strong>LEGITIMATE</strong></p>
                        <p><strong>Fraud Probability: {fraud_prob:.2%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fraud_prob * 100,
                    title={'text': "Fraud Risk Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if fraud_prob > 0.5 else "darkgreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Save to database
            if st.session_state.db_connected:
                prediction_data = {
                    'claim_details': input_data.to_dict('records')[0],
                    'prediction': int(is_fraud),
                    'fraud_probability': float(fraud_prob),
                    'model_used': st.session_state.model.best_model_name
                }
                st.session_state.db_manager.insert_prediction(prediction_data)
    
    with tab2:
        st.subheader("Batch Prediction from CSV")
        
        uploaded_file = st.file_uploader("Upload CSV file with claims data", type=['csv'])
        
        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)
            st.dataframe(batch_data.head())
            
            if st.button("🔮 Predict All"):
                with st.spinner("Making predictions..."):
                    predictions, probabilities = st.session_state.model.predict(batch_data)
                    
                    # Add results to dataframe
                    batch_data['fraud_prediction'] = predictions
                    batch_data['fraud_probability'] = probabilities
                    batch_data['risk_level'] = pd.cut(
                        probabilities,
                        bins=[0, 0.3, 0.7, 1.0],
                        labels=['Low', 'Medium', 'High']
                    )
                    
                    st.success("✅ Predictions completed!")
                    st.dataframe(batch_data)
                    
                    # Download results
                    csv = batch_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Summary
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Claims", len(batch_data))
                    
                    with col2:
                        st.metric("Fraud Detected", (predictions == 1).sum())
                    
                    with col3:
                        st.metric("Avg Risk Score", f"{probabilities.mean():.2%}")


def show_analytics_page():
    """Analytics dashboard page"""
    
    st.header("📈 Analytics Dashboard")
    
    if not st.session_state.db_connected:
        st.error("Database not connected!")
        return
    
    # Get predictions data
    predictions_df = st.session_state.db_manager.get_all_predictions()
    
    if predictions_df.empty:
        st.info("No prediction data available yet. Make some predictions first!")
        return
    
    # Overall statistics
    st.subheader("📊 Overall Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(predictions_df))
    
    with col2:
        fraud_count = predictions_df['prediction'].sum()
        st.metric("Fraud Cases", fraud_count)
    
    with col3:
        fraud_rate = (fraud_count / len(predictions_df)) * 100
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
    
    with col4:
        avg_prob = predictions_df['fraud_probability'].mean()
        st.metric("Avg Risk Score", f"{avg_prob:.2%}")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud vs Non-Fraud pie chart
        fraud_counts = predictions_df['prediction'].value_counts()
        
        # Handle case where only one type of prediction exists
        if len(fraud_counts) == 1:
            # If only one type, add the other with 0
            if 0 in fraud_counts.index:
                fraud_counts[1] = 0
            else:
                fraud_counts[0] = 0
            fraud_counts = fraud_counts.sort_index()
        
        # Map 0/1 to labels
        labels = ['Legitimate' if i == 0 else 'Fraudulent' for i in fraud_counts.index]
        
        fig = px.pie(
            values=fraud_counts.values,
            names=labels,
            title='Fraud Distribution',
            color_discrete_sequence=['#4caf50', '#f44336']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk score distribution
        fig = px.histogram(
            predictions_df,
            x='fraud_probability',
            nbins=30,
            title='Risk Score Distribution',
            labels={'fraud_probability': 'Fraud Probability'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series if prediction_date exists
    if 'prediction_date' in predictions_df.columns:
        predictions_df['prediction_date'] = pd.to_datetime(predictions_df['prediction_date'])
        predictions_df['date'] = predictions_df['prediction_date'].dt.date
        
        daily_counts = predictions_df.groupby(['date', 'prediction']).size().unstack(fill_value=0)
        
        fig = px.line(
            daily_counts,
            title='Daily Fraud Predictions Trend',
            labels={'value': 'Count', 'date': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)


def show_about_page():
    """About page"""
    
    st.header("ℹ️ About the Project")
    
    st.markdown("""
    ## 🧠 AI-Powered Insurance Fraud Detection System
    
    ### 📌 Project Overview
    The AI-Powered Insurance Fraud Detection System is a machine learning-based solution designed to detect 
    fraudulent insurance claims by analyzing claim details and historical transaction patterns.
    
    Insurance fraud leads to significant financial losses globally. This system leverages advanced machine 
    learning algorithms to classify claims as fraudulent or legitimate, enabling insurance companies to 
    reduce risks, improve investigation efficiency, and enhance decision-making accuracy.
    
    ### 🎯 Key Objectives
    - Develop a robust fraud classification model
    - Perform claim pattern analysis
    - Generate risk scoring for each claim
    - Provide decision support insights
    - Evaluate model performance using key metrics
    
    ### 🛠️ Technologies Used
    - **Frontend**: Streamlit
    - **Backend**: Python
    - **Database**: MongoDB
    - **ML Libraries**: Scikit-learn, XGBoost
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Data Processing**: Pandas, NumPy
    
    ### 📊 Features
    - **Fraud Classification**: Binary classification (Fraudulent/Legitimate)
    - **Pattern Analysis**: Identifies suspicious claim patterns
    - **Risk Scoring**: Probability-based fraud risk assessment
    - **Model Comparison**: Multiple ML algorithms (Random Forest, Decision Tree, XGBoost)
    - **Real-time Predictions**: Instant fraud detection
    - **Batch Processing**: Analyze multiple claims at once
    - **Analytics Dashboard**: Visual insights and trends
    
    ### 🏆 Innovation Aspect
    Unlike traditional rule-based systems, this solution uses machine learning to detect hidden fraud 
    patterns, continuously improving detection accuracy as more data becomes available.
    
    ### 👨‍💻 Developer
    Created for SmartInternz Hackathon
    
    ### 📄 License
    This project is created for educational and demonstration purposes.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📞 Support
    For questions or issues, please contact the development team.
    """)


if __name__ == "__main__":
    main()