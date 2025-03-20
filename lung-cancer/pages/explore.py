import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# ----------------------------- #
#        Page Configuration     #
# ----------------------------- #
st.set_page_config(
    page_title="Data Exploration - Lung Cancer Risk Assessment",
    page_icon="📊",
    layout="wide"
)

# ----------------------------- #
#         Data Loading          #
# ----------------------------- #
@st.cache_data
def load_dataset() -> pd.DataFrame:
    """
    Loads the lung cancer dataset from a CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        return pd.read_csv('lung_cancer.csv')
    except FileNotFoundError:
        return pd.read_csv('Lung Cancer Dataset.csv')

df = load_dataset()

# ----------------------------- #
#      Target Variable Setup    #
# ----------------------------- #
if 'PULMONARY_DISEASE' in df.columns:
    if pd.api.types.is_numeric_dtype(df['PULMONARY_DISEASE']):
        df['PULMONARY_DISEASE_NUMERIC'] = df['PULMONARY_DISEASE']
        # We'll re-map to display values after filtering
        target_col = 'PULMONARY_DISEASE_NUMERIC'
    else:
        df['PULMONARY_DISEASE_NUMERIC'] = df['PULMONARY_DISEASE'].map({'YES': 1, 'NO': 0})
        target_col = 'PULMONARY_DISEASE_NUMERIC'

# ----------------------------- #
#         Sidebar Filters       #
# ----------------------------- #
st.sidebar.header("Filter Data")
if 'AGE' in df.columns:
    min_age = int(df['AGE'].min())
    max_age = int(df['AGE'].max())
    age_range = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))
    df = df[(df['AGE'] >= age_range[0]) & (df['AGE'] <= age_range[1])]

if 'GENDER' in df.columns:
    if pd.api.types.is_numeric_dtype(df['GENDER']):
        gender_map = {0: "Female", 1: "Male"}
        df["GENDER_str"] = df["GENDER"].map(gender_map)
        gender_options = df["GENDER_str"].unique().tolist()
        selected_gender = st.sidebar.multiselect("Select Gender(s)", gender_options, default=gender_options)
        df = df[df["GENDER_str"].isin(selected_gender)]
    else:
        gender_options = df['GENDER'].unique().tolist()
        selected_gender = st.sidebar.multiselect("Select Gender(s)", gender_options, default=gender_options)
        df = df[df['GENDER'].isin(selected_gender)]

# Recalculate target display for filtered data
if 'PULMONARY_DISEASE' in df.columns:
    # Map numeric to YES/NO
    target_display_full = df[target_col].map({1: 'YES', 0: 'NO'})

# ----------------------------- #
#        Page Title & Intro     #
# ----------------------------- #
st.title("📊 Data Exploration: Understanding Lung Cancer Risk Factors")

# ----------------------------- #
#       Dataset Overview        #
# ----------------------------- #
st.header("Dataset Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Number of Records", f"{df.shape[0]}")
with col2:
    st.metric("Number of Features", f"{df.shape[1] - 1}")  # Excluding target
with col3:
    target_distribution = df[target_col].value_counts(normalize=True) * 100
    st.metric("Positive Cases", f"{target_distribution.get(1, 0):.1f}%")

with st.expander("View Sample Data & Details"):
    st.dataframe(df.head(10))
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data Types:**")
        st.write(df.dtypes)
    with col2:
        st.write("**Missing Values:**")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.write(missing_data[missing_data > 0])
        else:
            st.write("No missing values")

# ----------------------------- #
#       Target Distribution     #
# ----------------------------- #
st.header("Disease Distribution")
col1, col2 = st.columns(2)
with col1:
    value_counts = df[target_col].value_counts()
    if not value_counts.empty:
        fig_pie = px.pie(
            values=value_counts.values,
            names=['Negative', 'Positive'],
            title='Pulmonary Disease Case Distribution',
            color_discrete_sequence=px.colors.qualitative.Safe,
            hole=0.4
        )
        fig_pie.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.write("No data available for the selected filters.")
with col2:
    if not value_counts.empty:
        positive_count = df[target_col].sum()
        negative_count = len(df) - positive_count
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=['Negative', 'Positive'],
            y=[negative_count, positive_count],
            text=[f"{negative_count} ({negative_count/len(df)*100:.1f}%)", 
                  f"{positive_count} ({positive_count/len(df)*100:.1f}%)"],
            textposition='auto',
            marker_color=['#4ECDC4', '#FF6B6B']
        ))
        fig_bar.update_layout(
            title='Count of Pulmonary Disease Cases',
            xaxis_title='Disease Status',
            yaxis_title='Count'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.write("No data available for the selected filters.")

# ----------------------------- #
#      Interactive Feature Analysis   #
# ----------------------------- #
st.header("Feature Analysis")
feature_options = [col for col in df.columns if col not in ['PULMONARY_DISEASE', 'PULMONARY_DISEASE_NUMERIC', 'GENDER_str']]
selected_feature = st.selectbox("Select a feature to analyze:", feature_options)
is_categorical = (df[selected_feature].nunique() <= 5)

col1, col2 = st.columns(2)
with col1:
    if is_categorical:
        fig_feat = px.histogram(
            df,
            x=selected_feature,
            color=target_display_full,
            barmode='group',
            title=f'{selected_feature} Distribution by Disease Status',
            color_discrete_map={'YES': '#FF6B6B', 'NO': '#4ECDC4'}
        )
    else:
        fig_feat = px.histogram(
            df,
            x=selected_feature,
            color=target_display_full,
            marginal="box",
            title=f'{selected_feature} Distribution by Disease Status',
            color_discrete_map={'YES': '#FF6B6B', 'NO': '#4ECDC4'}
        )
    st.plotly_chart(fig_feat, use_container_width=True)
    
with col2:
    if is_categorical:
        cross_tab = pd.crosstab(df[selected_feature], df[target_col], normalize='index') * 100
        fig_crosstab = px.bar(
            cross_tab,
            barmode='group',
            title=f'Percentage of Disease by {selected_feature}',
            labels={'value': 'Percentage (%)', 'index': selected_feature}
        )
        st.plotly_chart(fig_crosstab, use_container_width=True)
        st.write(f"**{selected_feature} - Positive Case %:**")
        for val in sorted(df[selected_feature].unique()):
            positive_pct = cross_tab.loc[val, 1] if 1 in cross_tab.columns else 0
            st.write(f"- {selected_feature} = {val}: {positive_pct:.1f}%")
    else:
        stats = df.groupby(target_col)[selected_feature].agg(['mean', 'median', 'std']).reset_index()
        stats[target_col] = stats[target_col].map({1: 'Positive', 0: 'Negative'})
        st.write(f"**Statistics of {selected_feature} by Disease Status:**")
        st.dataframe(stats)
        fig_box = px.box(
            df,
            x=target_display_full,
            y=selected_feature,
            color=target_display_full,
            title=f'Box Plot of {selected_feature} by Disease Status',
            color_discrete_map={'YES': '#FF6B6B', 'NO': '#4ECDC4'}
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ----------------------------- #
#     Correlation Analysis      #
# ----------------------------- #
st.header("Correlation Analysis")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numeric_cols].corr()

tab1, tab2 = st.tabs(["Feature-Target Correlation", "Full Correlation Matrix"])
with tab1:
    target_corrs = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    fig_corr = px.bar(
        x=target_corrs.index,
        y=target_corrs.values,
        title='Correlation of Features with Pulmonary Disease',
        labels={'x': 'Feature', 'y': 'Correlation Coefficient'},
        color=target_corrs.values,
        color_continuous_scale='RdBu_r'
    )
    fig_corr.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown(
        """
        **Interpretation:**
        - **Positive correlation:** Feature increases with higher disease risk.
        - **Negative correlation:** Feature decreases with higher disease risk.
        """
    )
    
with tab2:
    fig_heat = px.imshow(
        corr_matrix,
        title='Feature Correlation Matrix',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ----------------------------- #
#      Risk Profiles Analysis   #
# ----------------------------- #
st.header("Risk Profiles Analysis")
top_risk_features = target_corrs.head(5).index.tolist()
if top_risk_features:
    risk_df = df[top_risk_features].copy()
    if risk_df.empty:
        st.write("No data available for risk profile analysis with the current filters.")
    else:
        scaler = StandardScaler()
        risk_df_scaled = pd.DataFrame(scaler.fit_transform(risk_df), columns=risk_df.columns)
        
        for col in risk_df_scaled.columns:
            risk_df_scaled[col] *= abs(target_corrs[col])
        df['risk_score'] = risk_df_scaled.sum(axis=1)
        
        df['risk_category'] = pd.qcut(df['risk_score'], q=3, labels=['Low Risk', 'Medium Risk', 'High Risk'])
        
        risk_vs_outcome = pd.crosstab(df['risk_category'], df[target_col], normalize='index') * 100
        fig_risk = px.bar(
            risk_vs_outcome,
            barmode='group',
            title='Disease Percentage by Risk Category',
            labels={'value': 'Percentage (%)', 'index': 'Risk Category'}
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        st.subheader("Risk Profile Characteristics")
        profile_tabs = st.tabs(['Low Risk', 'Medium Risk', 'High Risk'])
        for i, risk_level in enumerate(['Low Risk', 'Medium Risk', 'High Risk']):
            with profile_tabs[i]:
                risk_group = df[df['risk_category'] == risk_level]
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Sample Size:** {len(risk_group)}")
                    st.write(f"**Positive Cases:** {risk_group[target_col].mean() * 100:.1f}%")
                    st.write("**Key Characteristics:**")
                    for feature in top_risk_features:
                        if df[feature].nunique() <= 5:
                            most_common = risk_group[feature].value_counts(normalize=True).idxmax()
                            pct = risk_group[feature].value_counts(normalize=True).max() * 100
                            st.write(f"- {feature}: {most_common} ({pct:.1f}%)")
                        else:
                            avg_val = risk_group[feature].mean()
                            overall_avg = df[feature].mean()
                            comp = "higher" if avg_val > overall_avg else "lower"
                            st.write(f"- {feature}: Avg = {avg_val:.2f} ({comp} than overall)")
                with col2:
                    categories = top_risk_features.copy()
                    risk_avgs = []
                    overall_avgs = []
                    for feature in categories:
                        f_min = df[feature].min()
                        f_max = df[feature].max()
                        if f_min == f_max:
                            risk_avgs.append(0.5)
                            overall_avgs.append(0.5)
                        else:
                            risk_avgs.append((risk_group[feature].mean() - f_min) / (f_max - f_min))
                            overall_avgs.append((df[feature].mean() - f_min) / (f_max - f_min))
                    categories += [categories[0]]
                    risk_avgs += [risk_avgs[0]]
                    overall_avgs += [overall_avgs[0]]
                    fig_radar = px.line_polar(
                        r=risk_avgs, theta=categories, line_close=True,
                        title=f"Risk Factor Profile: {risk_level} vs. Overall"
                    )
                    fig_radar.add_scatterpolar(r=risk_avgs, theta=categories, fill='toself', name=f'{risk_level} Profile')
                    fig_radar.add_scatterpolar(r=overall_avgs, theta=categories, fill='toself', name='Overall Average')
                    st.plotly_chart(fig_radar, use_container_width=True)

# ----------------------------- #
#         Key Insights          #
# ----------------------------- #
st.header("Key Insights")
st.markdown(
    """
1. **Age and Smoking:** Older age and smoking are strongly associated with higher pulmonary disease risk.
2. **Respiratory Symptoms:** Breathing issues, chest tightness, and throat discomfort are key indicators.
3. **Family History:** Genetic factors moderately influence risk.
4. **Compound Effects:** Interaction effects (e.g., smoking with age) amplify overall risk.
5. **Risk Profiles:** Composite risk scores reveal distinct low, medium, and high risk groups, which can guide targeted interventions.
    """
)

# ----------------------------- #
#        Navigation Links       #
# ----------------------------- #
st.markdown("---")
col_nav1, col_nav2 = st.columns(2)
with col_nav1:
    st.page_link("app.py", label="← Back to Home", icon="🏠")
with col_nav2:
    st.page_link("pages/predict.py", label="Get Your Risk Assessment →", icon="🔍")
