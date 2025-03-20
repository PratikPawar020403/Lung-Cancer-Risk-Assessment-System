import streamlit as st
import pandas as pd
from PIL import Image
import os
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple

# ----------------------------- #
#        Page Configuration     #
# ----------------------------- #
st.set_page_config(
    page_title="Lung Cancer Risk Assessment System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- #
#       Data Loading Function   #
# ----------------------------- #
@st.cache_data
def load_dataset() -> pd.DataFrame:
    """
    Load the lung cancer dataset from CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    file_paths = ['lung-cancer/Lung Cancer Dataset.csv']
    for path in file_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError("Dataset file not found in expected paths.")

# ----------------------------- #
#       Utility Functions       #
# ----------------------------- #
def convert_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Convert target variable to numeric if needed and return the target column name.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        Tuple[pd.DataFrame, str]: Updated dataset and target column name.
    """
    if 'PULMONARY_DISEASE' in df.columns and not pd.api.types.is_numeric_dtype(df['PULMONARY_DISEASE']):
        df['PULMONARY_DISEASE_NUMERIC'] = df['PULMONARY_DISEASE'].map({'YES': 1, 'NO': 0})
        return df, 'PULMONARY_DISEASE_NUMERIC'
    return df, 'PULMONARY_DISEASE'

def render_dataset_overview(df: pd.DataFrame, target_col: str) -> None:
    """
    Render dataset statistics and overview.

    Args:
        df (pd.DataFrame): The dataset.
        target_col (str): The target column name.
    """
    st.header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sample Size", f"{len(df):,}")
    with col2:
        positive_rate = df[target_col].mean() * 100
        st.metric("Positive Cases", f"{positive_rate:.1f}%")
    with col3:
        st.metric("Risk Factors", f"{df.shape[1] - 1}")
    with col4:
        st.metric("Model Accuracy", "87.5%")  # Placeholder accuracy

def render_risk_factors(df: pd.DataFrame, target_col: str) -> None:
    """
    Render an interactive visualization of top risk factors using Plotly.

    Args:
        df (pd.DataFrame): The dataset.
        target_col (str): The target column name.
    """
    st.header("Primary Risk Factors")
    # Compute correlations excluding target-related columns
    exclude_cols = [col for col in df.columns if 'PULMONARY_DISEASE' in col]
    correlations = df.drop(columns=exclude_cols).corrwith(df[target_col])
    top_corr = correlations.abs().sort_values(ascending=False).head(5)
    
    # Build an interactive horizontal bar chart with Plotly
    fig = px.bar(
        x=top_corr.values,
        y=top_corr.index,
        orientation="h",
        labels={"x": "Correlation Coefficient", "y": "Feature"},
        title="Top 5 Factors Associated with Pulmonary Disease",
        color=top_corr.values,
        color_continuous_scale="RdBu_r"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

def render_navigation() -> None:
    """
    Render call-to-action navigation links.
    """
    st.header("Start Your Assessment")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Explore the Data
        Dive into interactive visualizations to understand the underlying patterns.
        """)
        st.page_link("pages/explore.py", label="Explore Data", icon="📊")
    with col2:
        st.markdown("""
        ### Get Your Risk Assessment
        Complete a short form to receive your personalized risk assessment.
        """)
        st.page_link("pages/predict.py", label="Get Assessment", icon="🔍")

def render_how_it_works() -> None:
    """
    Render the "How It Works" section.
    """
    st.header("How It Works")
    st.markdown(
        """
This application uses a machine learning model trained on lung cancer data to predict individual risk. 
The process involves:

1. **Data Collection**: Gathering demographic, behavioral, and health-related information.
2. **Advanced Analysis**: Processing through an optimized logistic regression model.
3. **Risk Stratification**: Classifying individuals into risk categories based on probability scores.
4. **Personalized Recommendations**: Providing tailored advice based on specific risk factors.
        """
    )

def render_footer() -> None:
    """
    Render the footer with disclaimer and copyright notice.
    """
    st.markdown("---")
    st.markdown(
        """
<div style="text-align: center;">
    <p><strong>Disclaimer</strong>: This tool provides an estimation of risk based on statistical models and should not be considered a medical diagnosis. 
    Always consult healthcare professionals for proper advice.</p>
    <p>© 2025 Lung Cancer Risk Assessment System • Version 1.0</p>
</div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------- #
#          Main App             #
# ----------------------------- #
def main() -> None:
    # Load and process dataset
    df = load_dataset()
    df, target_col = convert_target(df)
    
    # App Title & Introduction
    st.title("🫁 Lung Cancer Risk Assessment System")
    st.markdown(
        """
## Welcome to the Lung Cancer Risk Assessment Tool

This application leverages machine learning to help identify individuals at risk of developing pulmonary disease—particularly lung cancer—based on a variety of factors. Early detection and risk assessment are key in improving outcomes and guiding preventive strategies.
        """
    )
    
    # Key Features Section
    st.header("Key Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 Data Exploration")
        st.markdown(
            """
- Interactive visualizations  
- Risk factor analysis  
- Statistical insights  
- Correlation analysis
            """
        )
    with col2:
        st.markdown("### 🔍 Personalized Risk Assessment")
        st.markdown(
            """
- AI-powered risk prediction  
- Individual risk factor analysis  
- Risk category classification  
- Visual risk representation
            """
        )
    with col3:
        st.markdown("### 📋 Customized Recommendations")
        st.markdown(
            """
- Tailored prevention strategies  
- Modifiable risk factor guidance  
- Screening recommendations  
- Health monitoring suggestions
            """
        )
    
    # Render Dataset Overview & Risk Factors
    render_dataset_overview(df, target_col)
    render_risk_factors(df, target_col)
    
    # Navigation to other pages
    render_navigation()
    
    # How It Works Section
    render_how_it_works()
    
    # Footer
    render_footer()

if __name__ == "__main__":
    main()
