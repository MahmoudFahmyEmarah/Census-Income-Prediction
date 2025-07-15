"""
Census Income Prediction - Streamlit Application
A comprehensive data science web application for income prediction using machine learning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from model_loader import get_model_loader

# Configure page
st.set_page_config(
    page_title="Census Income Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-result {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .high-income {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
    }
    
    .low-income {
        background: linear-gradient(135deg, #2196F3, #1976D2);
        color: white;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_data
def load_model_data():
    """Load model performance data and sample predictions"""
    model_loader = get_model_loader()
    model_performance = model_loader.get_model_performance()
    data_stats = model_loader.get_data_stats()
    return model_performance, data_stats

def predict_income(input_data):
    """Make prediction using trained models"""
    model_loader = get_model_loader()
    return model_loader.predict(input_data)

def main():
    # Sidebar navigation
    st.sidebar.markdown("# 🧠 Census Income AI")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["🏠 Home Dashboard", "🎯 Live Predictions", "📊 Model Comparison", 
         "📈 Data Insights", "🔍 Feature Importance", "ℹ️ About"]
    )
    
    # Load data
    model_performance, data_stats = load_model_data()
    
    if page == "🏠 Home Dashboard":
        show_dashboard(model_performance, data_stats)
    elif page == "🎯 Live Predictions":
        show_prediction_interface()
    elif page == "📊 Model Comparison":
        show_model_comparison(model_performance)
    elif page == "📈 Data Insights":
        show_data_insights(data_stats)
    elif page == "🔍 Feature Importance":
        show_feature_importance()
    elif page == "ℹ️ About":
        show_about()

def show_dashboard(model_performance, data_stats):
    """Display the main dashboard"""
    st.markdown('<h1 class="main-header">Census Income Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Best Accuracy</h3>
            <h2>96.32%</h2>
            <p>XGBoost Model</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 ROC-AUC</h3>
            <h2>99.30%</h2>
            <p>Exceptional Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Training Data</h3>
            <h2>300K+</h2>
            <p>Census Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>⚙️ Features</h3>
            <h2>188</h2>
            <p>Engineered Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model performance chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Model Performance Comparison")
        
        models = list(model_performance.keys())
        accuracies = [model_performance[model]['accuracy'] for model in models]
        
        fig = px.bar(
            x=models, 
            y=accuracies,
            title="Model Accuracy Comparison",
            labels={'x': 'Model', 'y': 'Accuracy (%)'},
            color=accuracies,
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 ROC-AUC Scores")
        
        roc_scores = [model_performance[model]['roc_auc'] for model in models]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=models,
            y=roc_scores,
            mode='lines+markers',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10, color='#764ba2')
        ))
        fig.update_layout(
            title="ROC-AUC Performance",
            xaxis_title="Model",
            yaxis_title="ROC-AUC Score (%)",
            yaxis=dict(range=[90, 100])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data overview
    st.subheader("📈 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Samples", f"{data_stats['training_samples']:,}")
        st.metric("Test Samples", f"{data_stats['test_samples']:,}")
    
    with col2:
        st.metric("Original Features", data_stats['original_features'])
        st.metric("Engineered Features", data_stats['engineered_features'])
    
    with col3:
        st.metric("Duplicates Removed", f"{data_stats['duplicates_removed']:,}")
        st.metric("Feature Increase", f"{((data_stats['engineered_features'] / data_stats['original_features']) - 1) * 100:.0f}%")
    
    # Class distribution
    st.subheader("⚖️ Income Class Distribution")
    
    labels = ['Low Income (≤$50K)', 'High Income (>$50K)']
    values = [data_stats['class_imbalance']['low_income'], data_stats['class_imbalance']['high_income']]
    
    fig = px.pie(
        values=values, 
        names=labels,
        title="Original Dataset Class Distribution",
        color_discrete_sequence=['#3498db', '#2ecc71']
    )
    st.plotly_chart(fig, use_container_width=True)

def show_prediction_interface():
    """Display the prediction interface"""
    st.markdown('<h1 class="main-header">🎯 Live Income Prediction</h1>', unsafe_allow_html=True)
    
    st.markdown("### Enter demographic information to predict income classification")
    
    # Create form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=16, max_value=100, value=35)
            
            education = st.selectbox("Education Level", [
                'Less than 1st grade',
                '1st 2nd 3rd or 4th grade',
                '5th or 6th grade',
                '7th and 8th grade',
                '9th grade',
                '10th grade',
                '11th grade',
                '12th grade no diploma',
                'High school graduate',
                'Some college but no degree',
                'Associates degree-occup /vocational',
                'Associates degree-academic program',
                'Bachelors degree(BA AB BS)',
                'Masters degree(MA MS MEng MEd MSW MBA)',
                'Prof school degree (MD DDS DVM LLB JD)',
                'Doctorate degree(PhD EdD)'
            ], index=8)
            
            marital_status = st.selectbox("Marital Status", [
                'Never married',
                'Married-civilian spouse present',
                'Married-spouse absent',
                'Separated',
                'Divorced',
                'Widowed'
            ])
            
            sex = st.selectbox("Sex", ['Male', 'Female'])
            
            race = st.selectbox("Race", [
                'White',
                'Black',
                'Asian or Pacific Islander',
                'Amer Indian Aleut or Eskimo',
                'Other'
            ])
        
        with col2:
            occupation_code = st.selectbox("Occupation", [
                'Professional specialty',
                'Executive admin and managerial',
                'Sales',
                'Other service',
                'Precision production craft & repair',
                'Machine operators assmblrs & inspctrs',
                'Transportation and material moving',
                'Handlers equip cleaners etc',
                'Farming forestry and fishing',
                'Technicians and related support',
                'Protective services',
                'Private household services',
                'Adm support including clerical',
                'Armed Forces'
            ])
            
            class_of_worker = st.selectbox("Class of Worker", [
                'Private',
                'Self-employed-not incorporated',
                'Local government',
                'State government',
                'Federal government',
                'Self-employed-incorporated',
                'Without pay',
                'Never worked'
            ])
            
            capital_gains = st.number_input("Capital Gains", min_value=0, value=0)
            capital_losses = st.number_input("Capital Losses", min_value=0, value=0)
            weeks_worked_in_year = st.number_input("Weeks Worked in Year", min_value=0, max_value=52, value=40)
        
        # Submit button
        submitted = st.form_submit_button("🧠 Predict Income", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = {
                'age': age,
                'education': education,
                'marital_status': marital_status,
                'occupation_code': occupation_code,
                'sex': sex,
                'race': race,
                'class_of_worker': class_of_worker,
                'capital_gains': capital_gains,
                'capital_losses': capital_losses,
                'weeks_worked_in_year': weeks_worked_in_year
            }
            
            # Make prediction
            result = predict_income(input_data)
            
            # Display result
            if result['prediction'] == 1:
                st.markdown(f"""
                <div class="prediction-result high-income">
                    🎉 HIGH INCOME (>$50K)
                    <br>
                    <small>Confidence: {result['confidence']*100:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-result low-income">
                    💼 LOW INCOME (≤$50K)
                    <br>
                    <small>Confidence: {result['confidence']*100:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Show probability breakdown
            st.subheader("📊 Prediction Breakdown")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("High Income Probability", f"{result['probability_high_income']*100:.1f}%")
            with col2:
                st.metric("Low Income Probability", f"{result['probability_low_income']*100:.1f}%")
            
            # Probability chart
            fig = go.Figure(data=[
                go.Bar(name='Probabilities', 
                      x=['Low Income', 'High Income'], 
                      y=[result['probability_low_income']*100, result['probability_high_income']*100],
                      marker_color=['#3498db', '#2ecc71'])
            ])
            fig.update_layout(title="Income Prediction Probabilities", yaxis_title="Probability (%)")
            st.plotly_chart(fig, use_container_width=True)
    
    # Sample data section
    st.markdown("---")
    st.subheader("🚀 Quick Test with Sample Data")
    
    sample_profiles = {
        "Young Professional": {
            'age': 28, 'education': 'Bachelors degree(BA AB BS)', 'marital_status': 'Never married',
            'occupation_code': 'Professional specialty', 'sex': 'Male', 'race': 'White',
            'class_of_worker': 'Private', 'capital_gains': 0, 'capital_losses': 0, 'weeks_worked_in_year': 52
        },
        "Senior Executive": {
            'age': 45, 'education': 'Masters degree(MA MS MEng MEd MSW MBA)', 'marital_status': 'Married-civilian spouse present',
            'occupation_code': 'Executive admin and managerial', 'sex': 'Male', 'race': 'White',
            'class_of_worker': 'Private', 'capital_gains': 5178, 'capital_losses': 0, 'weeks_worked_in_year': 52
        },
        "Part-time Worker": {
            'age': 22, 'education': 'High school graduate', 'marital_status': 'Never married',
            'occupation_code': 'Other service', 'sex': 'Female', 'race': 'White',
            'class_of_worker': 'Private', 'capital_gains': 0, 'capital_losses': 0, 'weeks_worked_in_year': 30
        }
    }
    
    col1, col2, col3 = st.columns(3)
    
    for i, (profile_name, profile_data) in enumerate(sample_profiles.items()):
        col = [col1, col2, col3][i]
        with col:
            st.markdown(f"**{profile_name}**")
            st.write(f"Age: {profile_data['age']}, {profile_data['sex']}")
            st.write(f"Education: {profile_data['education']}")
            st.write(f"Occupation: {profile_data['occupation_code']}")
            
            if st.button(f"Test {profile_name}", key=f"sample_{i}"):
                result = predict_income(profile_data)
                prediction_text = "HIGH INCOME" if result['prediction'] == 1 else "LOW INCOME"
                st.success(f"Prediction: {prediction_text} ({result['confidence']*100:.1f}% confidence)")

def show_model_comparison(model_performance):
    """Display model comparison"""
    st.markdown('<h1 class="main-header">📊 Model Comparison</h1>', unsafe_allow_html=True)
    
    # Create comparison dataframe
    df = pd.DataFrame(model_performance).T
    df = df.round(2)
    
    # Display metrics table
    st.subheader("📋 Performance Metrics")
    st.dataframe(df, use_container_width=True)
    
    # Metrics comparison charts
    metrics = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']
    
    for i in range(0, len(metrics), 2):
        cols = st.columns(2)
        
        for j, col in enumerate(cols):
            if i + j < len(metrics):
                metric = metrics[i + j]
                
                with col:
                    fig = px.bar(
                        x=list(model_performance.keys()),
                        y=[model_performance[model][metric] for model in model_performance.keys()],
                        title=f"{metric.replace('_', ' ').title()} Comparison",
                        labels={'x': 'Model', 'y': f'{metric.replace("_", " ").title()} (%)'},
                        color=[model_performance[model][metric] for model in model_performance.keys()],
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Radar chart for comprehensive comparison
    st.subheader("🎯 Comprehensive Model Comparison")
    
    fig = go.Figure()
    
    for model in model_performance.keys():
        fig.add_trace(go.Scatterpolar(
            r=[model_performance[model][metric] for metric in metrics],
            theta=[metric.replace('_', ' ').title() for metric in metrics],
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[80, 100]
            )),
        showlegend=True,
        title="Model Performance Radar Chart"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_data_insights(data_stats):
    """Display data insights"""
    st.markdown('<h1 class="main-header">📈 Data Insights</h1>', unsafe_allow_html=True)
    
    # Dataset statistics
    st.subheader("📊 Dataset Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Samples", f"{data_stats['training_samples']:,}")
        st.metric("Test Samples", f"{data_stats['test_samples']:,}")
    
    with col2:
        st.metric("Original Features", data_stats['original_features'])
        st.metric("Engineered Features", data_stats['engineered_features'])
    
    with col3:
        st.metric("Duplicates Removed", f"{data_stats['duplicates_removed']:,}")
        improvement = ((data_stats['engineered_features'] / data_stats['original_features']) - 1) * 100
        st.metric("Feature Engineering Improvement", f"+{improvement:.0f}%")
    
    # Class distribution
    st.subheader("⚖️ Class Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        labels = ['Low Income (≤$50K)', 'High Income (>$50K)']
        values = [data_stats['class_imbalance']['low_income'], data_stats['class_imbalance']['high_income']]
        
        fig = px.pie(
            values=values, 
            names=labels,
            title="Income Distribution",
            color_discrete_sequence=['#3498db', '#2ecc71']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart
        fig = px.bar(
            x=labels,
            y=values,
            title="Class Imbalance",
            labels={'x': 'Income Class', 'y': 'Percentage (%)'},
            color=values,
            color_continuous_scale='RdYlBu'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("🔍 Key Data Insights")
    
    insights = [
        "📊 **Severe Class Imbalance**: 93.8% low income vs 6.2% high income - required advanced balancing techniques",
        "🔧 **Extensive Feature Engineering**: Increased features from 40 to 188 (370% improvement)",
        "🧹 **Data Quality**: Removed 78,273 duplicate records for cleaner training data",
        "📈 **Large Scale**: Processed 300K+ census records for robust model training",
        "⚖️ **Balanced Training**: Used SMOTE and undersampling for optimal model performance",
        "🎯 **High Quality**: Achieved 99.3% ROC-AUC through comprehensive preprocessing"
    ]
    
    for insight in insights:
        st.markdown(insight)
    
    # Processing pipeline
    st.subheader("🔄 Data Processing Pipeline")
    
    pipeline_steps = [
        "Raw Data Loading",
        "Duplicate Removal",
        "Missing Value Handling",
        "Feature Engineering",
        "Class Balancing",
        "Model Training"
    ]
    
    # Create a simple flow chart
    fig = go.Figure()
    
    for i, step in enumerate(pipeline_steps):
        fig.add_trace(go.Scatter(
            x=[i], y=[0],
            mode='markers+text',
            marker=dict(size=50, color='lightblue'),
            text=step,
            textposition="middle center",
            showlegend=False
        ))
        
        if i < len(pipeline_steps) - 1:
            fig.add_trace(go.Scatter(
                x=[i, i+1], y=[0, 0],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
    
    fig.update_layout(
        title="Data Processing Pipeline",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=200
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_feature_importance():
    """Display feature importance analysis"""
    st.markdown('<h1 class="main-header">🔍 Feature Importance Analysis</h1>', unsafe_allow_html=True)
    
    # Get feature importance from model loader
    model_loader = get_model_loader()
    feature_importance = model_loader.get_feature_importance()
    
    st.subheader("🏆 Top Features Driving Income Predictions")
    
    # Feature importance chart
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance (XGBoost Model)",
        labels={'x': 'Importance Score', 'y': 'Features'},
        color=importance,
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature categories
    st.subheader("📊 Feature Categories")
    
    categories = {
        'Education': ['education_level', 'has_college_degree'],
        'Demographics': ['age', 'sex', 'race', 'marital_status'],
        'Work': ['occupation_code', 'class_of_worker', 'weeks_worked_in_year'],
        'Financial': ['capital_gains', 'capital_losses', 'dividends_from_stocks'],
        'Engineered': ['age_education_interaction', 'is_married', 'work_intensity']
    }
    
    category_importance = {}
    for category, features_list in categories.items():
        total_importance = sum(feature_importance.get(feature, 0) for feature in features_list)
        category_importance[category] = total_importance
    
    fig = px.pie(
        values=list(category_importance.values()),
        names=list(category_importance.keys()),
        title="Feature Importance by Category"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎓 Education is King**
        - Education level is the strongest predictor (15% importance)
        - College degree significantly impacts income potential
        - Professional degrees show highest income correlation
        """)
        
        st.markdown("""
        **👥 Demographics Matter**
        - Age shows strong correlation with income
        - Marital status influences earning potential
        - Gender and race factors considered for fairness
        """)
    
    with col2:
        st.markdown("""
        **💼 Work Characteristics**
        - Occupation type is crucial (10% importance)
        - Full-time work correlates with higher income
        - Self-employment shows mixed results
        """)
        
        st.markdown("""
        **💰 Financial Indicators**
        - Capital gains are strong income predictors
        - Investment income indicates wealth
        - Financial diversity matters
        """)

def show_about():
    """Display about page"""
    st.markdown('<h1 class="main-header">ℹ️ About This Project</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This comprehensive data science project demonstrates advanced machine learning techniques for predicting income levels 
    using U.S. Census data. The application showcases the complete data science pipeline from raw data processing 
    to production deployment.
    
    ## 🏆 Key Achievements
    
    - **99.3% ROC-AUC Score** with XGBoost and LightGBM models
    - **96.32% Accuracy** on balanced test data
    - **188 Engineered Features** from 40 original features
    - **300K+ Training Samples** processed and cleaned
    - **Production-Ready Deployment** with Streamlit interface
    
    ## 🔬 Methodology
    
    ### Data Processing
    1. **Data Loading**: Processed 300K+ census records
    2. **Quality Assessment**: Identified and handled missing values, duplicates
    3. **Feature Engineering**: Created 148 new features through domain expertise
    4. **Class Balancing**: Applied SMOTE and undersampling techniques
    
    ### Model Development
    1. **Multiple Algorithms**: Logistic Regression, Random Forest, XGBoost, LightGBM, Neural Network
    2. **Cross-Validation**: 5-fold CV for robust performance estimation
    3. **Hyperparameter Tuning**: Grid search and random search optimization
    4. **Ensemble Methods**: Combined predictions for improved accuracy
    
    ### Evaluation & Validation
    1. **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
    2. **Bias Analysis**: Fairness evaluation across demographic groups
    3. **Feature Importance**: SHAP values for model interpretability
    4. **Production Testing**: Real-world performance validation
    
    ## 🛠️ Technical Stack
    
    - **Data Processing**: Python, Pandas, NumPy, Scikit-learn
    - **Machine Learning**: XGBoost, LightGBM, TensorFlow/Keras
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Web Application**: Streamlit
    - **Deployment**: Docker, Cloud platforms
    
    ## 📊 Dataset Information
    
    **Source**: U.S. Census Bureau  
    **Size**: 299,285 total records (199,523 training + 99,762 testing)  
    **Features**: 40 original demographic and economic features  
    **Target**: Binary income classification (≤$50K vs >$50K)  
    **Challenge**: Severe class imbalance (93.8% vs 6.2%)  
    
    ## 🎓 Business Applications
    
    - **Market Segmentation**: Identify high-value customer segments
    - **Credit Assessment**: Enhance loan approval processes
    - **HR Analytics**: Inform compensation and recruitment strategies
    - **Policy Analysis**: Support economic and social policy decisions
    - **Risk Management**: Assess financial risk profiles
    
    ## 🔮 Future Enhancements
    
    - **Real-time Predictions**: API integration for live predictions
    - **Advanced Models**: Deep learning and ensemble techniques
    - **Bias Mitigation**: Enhanced fairness algorithms
    - **Feature Selection**: Automated feature importance ranking
    - **Model Monitoring**: Performance tracking and drift detection
    
    ## 👨‍💻 Development Team
    
    This project was developed as a comprehensive demonstration of advanced data science and machine learning capabilities, 
    showcasing end-to-end project development from data exploration to production deployment.
    
    ---
    
    **Last Updated**: {datetime.now().strftime("%B %d, %Y")}  
    **Version**: 1.0.0  
    **Status**: Production Ready ✅
    """)

if __name__ == "__main__":
    main()

