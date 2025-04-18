import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="MindTrace Mental Health Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Load the dataset
@st.cache_data
def load_data():
    try:
        # Try to load the processed data
        df = pd.read_csv(r"C:\Users\DELL\Downloads\Machine learning projects\Mental Health Detection\processed_mental_health_data.csv")
    except FileNotFoundError:
        try:
            # If not available, load and process the raw data
            df = pd.read_csv("mental_health_tech_usage.csv")
            st.warning("Using raw data. For better results, run the preprocessing script first.")
            
            # Minimal processing for dashboard if raw data
            if 'Mental_Health_Status' in df.columns:
                # Map mental health status to numeric
                mental_health_mapping = {
                    'Excellent': 5, 'Good': 4, 'Fair': 3, 'Poor': 2, 'Very Poor': 1
                }
                df['Mental_Health_Score'] = df['Mental_Health_Status'].map(mental_health_mapping)
            
            if 'Stress_Level' in df.columns:
                # Map stress level to numeric
                stress_mapping = {
                    'Very High': 5, 'High': 4, 'Moderate': 3, 'Low': 2, 'Very Low': 1
                }
                df['Stress_Score'] = df['Stress_Level'].map(stress_mapping)
            
            # Age groups
            bins = [0, 18, 25, 35, 50, 65, 100]
            labels = ['Under 18', '18-25', '26-35', '36-50', '51-65', 'Over 65']
            df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
            
            # Screen time categories
            screen_bins = [0, 2, 4, 8, 24]
            screen_labels = ['Low (0-2hrs)', 'Moderate (2-4hrs)', 'High (4-8hrs)', 'Very High (8+hrs)']
            df['Screen_Time_Category'] = pd.cut(df['Screen_Time_Hours'], bins=screen_bins, labels=screen_labels, right=False)
            
            # Basic risk index
            if all(col in df.columns for col in ['Screen_Time_Hours', 'Mental_Health_Score', 'Stress_Score']):
                # Risk factors
                screen_risk = df['Screen_Time_Hours'] / 12
                mental_risk = (6 - df['Mental_Health_Score']) / 5
                stress_risk = (df['Stress_Score'] - 1) / 4
                
                # Combine risk factors
                df['Risk_Index'] = (screen_risk * 0.33 + mental_risk * 0.33 + stress_risk * 0.33) * 10
                
                # Risk categories
                conditions = [
                    (df['Risk_Index'] < 3),
                    (df['Risk_Index'] >= 3) & (df['Risk_Index'] < 5),
                    (df['Risk_Index'] >= 5) & (df['Risk_Index'] < 7),
                    (df['Risk_Index'] >= 7)
                ]
                values = ['Low Risk', 'Moderate Risk', 'High Risk', 'Severe Risk']
                df['Risk_Category'] = np.select(conditions, values, default='Unknown')
        except FileNotFoundError:
            # If no data available, create sample data
            st.error("Dataset not found. Please upload a dataset or use the preprocessing script.")
            df = pd.DataFrame()
    
    return df

# Main function
def main():
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #4A7BB7;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #4A7BB7;
            margin-top: 1rem;
        }
        .highlight {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 5px;
        }
        .risk-high {
            color: #FF4B4B;
            font-weight: bold;
        }
        .risk-moderate {
            color: #FFA500;
            font-weight: bold;
        }
        .risk-low {
            color: #00CC96;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">MindTrace: Mental Health & Technology Usage Dashboard</div>', unsafe_allow_html=True)
    st.markdown("""
        This dashboard explores the relationship between technology usage patterns and mental health indicators, 
        helping identify early warning signs of mental health issues through passive monitoring.
    """)
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.warning("No data available. Please check your dataset path.")
        return
    
    # Sidebar - Filters
    st.sidebar.header("Filters")
    
    # Age group filter
    if 'Age_Group' in df.columns:
        age_groups = sorted(df['Age_Group'].unique().tolist())
        selected_age_groups = st.sidebar.multiselect(
            "Select Age Groups",
            options=age_groups,
            default=age_groups
        )
    else:
        selected_age_groups = []
    
    # Gender filter
    if 'Gender' in df.columns:
        genders = sorted(df['Gender'].unique().tolist())
        selected_genders = st.sidebar.multiselect(
            "Select Genders",
            options=genders,
            default=genders
        )
    else:
        selected_genders = []
    
    # Screen time category filter
    if 'Screen_Time_Category' in df.columns:
        screen_categories = sorted(df['Screen_Time_Category'].unique().tolist())
        selected_screen_categories = st.sidebar.multiselect(
            "Select Screen Time Categories",
            options=screen_categories,
            default=screen_categories
        )
    else:
        selected_screen_categories = []
    
    # Risk category filter if available
    if 'Risk_Category' in df.columns:
        risk_categories = sorted(df['Risk_Category'].unique().tolist())
        selected_risk_categories = st.sidebar.multiselect(
            "Select Risk Categories",
            options=risk_categories,
            default=risk_categories
        )
    else:
        selected_risk_categories = []
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_age_groups and 'Age_Group' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Age_Group'].isin(selected_age_groups)]
    
    if selected_genders and 'Gender' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Gender'].isin(selected_genders)]
    
    if selected_screen_categories and 'Screen_Time_Category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Screen_Time_Category'].isin(selected_screen_categories)]
    
    if selected_risk_categories and 'Risk_Category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Risk_Category'].isin(selected_risk_categories)]
    
    # Show filter status
    st.sidebar.info(f"Showing {len(filtered_df)} out of {len(df)} records")
    
    # Reset filters button
    if st.sidebar.button("Reset All Filters"):
        st.experimental_rerun()
    
    # Main dashboard content
    # Top metrics
    st.markdown('<div class="sub-header">Key Technology Usage Metrics</div>', unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        if 'Screen_Time_Hours' in filtered_df.columns:
            avg_screen_time = filtered_df['Screen_Time_Hours'].mean()
            st.metric("Avg. Screen Time", f"{avg_screen_time:.2f} hrs")
    
    with metric_col2:
        if 'Social_Media_Usage_Hours' in filtered_df.columns:
            avg_social = filtered_df['Social_Media_Usage_Hours'].mean()
            st.metric("Avg. Social Media Usage", f"{avg_social:.2f} hrs")
    
    with metric_col3:
        if 'Gaming_Hours' in filtered_df.columns:
            avg_gaming = filtered_df['Gaming_Hours'].mean()
            st.metric("Avg. Gaming Time", f"{avg_gaming:.2f} hrs")
    
    # Mental health metrics
    st.markdown('<div class="sub-header">Key Mental Health Metrics</div>', unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    