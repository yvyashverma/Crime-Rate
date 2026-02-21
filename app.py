import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crime Rate EDA Dashboard",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .stat-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚔 Crime Rate Exploratory Data Analysis</h1>', unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('crime_data.csv')
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error("Crime data file not found. Please run generate_data.py first.")
        return None

df = load_data()

if df is not None:
    # Sidebar
    st.sidebar.header("🔍 Filters")
    
    # Date range filter
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df[(df['date'] >= pd.Timestamp(start_date)) & 
                         (df['date'] <= pd.Timestamp(end_date))]
    else:
        df_filtered = df.copy()
    
    # City filter
    cities = ['All'] + sorted(df['city'].unique().tolist())
    selected_city = st.sidebar.selectbox("Select City", cities)
    
    if selected_city != 'All':
        df_filtered = df_filtered[df_filtered['city'] == selected_city]
    
    # Crime type filter
    crime_types = ['All'] + sorted(df['crime_type'].unique().tolist())
    selected_crime = st.sidebar.selectbox("Select Crime Type", crime_types)
    
    if selected_crime != 'All':
        df_filtered = df_filtered[df_filtered['crime_type'] == selected_crime]
    
    # Severity filter
    severity_range = st.sidebar.slider(
        "Severity Range",
        min_value=int(df['severity'].min()),
        max_value=int(df['severity'].max()),
        value=(int(df['severity'].min()), int(df['severity'].max()))
    )
    df_filtered = df_filtered[(df_filtered['severity'] >= severity_range[0]) & 
                              (df_filtered['severity'] <= severity_range[1])]
    
    # KPI Metrics
    st.markdown("## 📊 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Crimes", f"{len(df_filtered):,}")
    with col2:
        st.metric("Unique Cities", df_filtered['city'].nunique())
    with col3:
        st.metric("Crime Types", df_filtered['crime_type'].nunique())
    with col4:
        arrest_rate = (df_filtered['arrested'].sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
        st.metric("Arrest Rate", f"{arrest_rate:.1f}%")
    with col5:
        avg_severity = df_filtered['severity'].mean()
        st.metric("Avg Severity", f"{avg_severity:.2f}")
    
    st.markdown("---")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Temporal Analysis", 
        "📍 Geographic Analysis", 
        "📊 Crime Patterns",
        "📋 Demographic Analysis",
        "📑 Summary Report"
    ])
    
    with tab1:
        st.markdown("### Temporal Analysis of Crimes")
        
        # Time series plot
        st.subheader("Crime Trends Over Time")
        crime_over_time = df_filtered.groupby(df_filtered['date'].dt.to_period('M')).size().reset_index(name='count')
        crime_over_time['date'] = crime_over_time['date'].astype(str)
        
        fig = px.line(crime_over_time, x='date', y='count', 
                      title='Monthly Crime Count',
                      labels={'count': 'Number of Crimes', 'date': 'Month'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Hourly and daily patterns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Crimes by Hour of Day")
            hourly_crimes = df_filtered['hour'].value_counts().sort_index()
            fig_hour = px.bar(x=hourly_crimes.index, y=hourly_crimes.values,
                            labels={'x': 'Hour', 'y': 'Number of Crimes'},
                            title='Hourly Crime Distribution')
            st.plotly_chart(fig_hour, use_container_width=True)
        
        with col2:
            st.subheader("Crimes by Day of Week")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_crimes = df_filtered['day_of_week'].value_counts().reindex(day_order)
            fig_day = px.bar(x=daily_crimes.index, y=daily_crimes.values,
                           labels={'x': 'Day of Week', 'y': 'Number of Crimes'},
                           title='Daily Crime Distribution')
            st.plotly_chart(fig_day, use_container_width=True)
        
        # Seasonal analysis
        st.subheader("Seasonal Crime Patterns")
        seasonal = df_filtered.groupby(['season', 'crime_type']).size().reset_index(name='count')
        fig_season = px.bar(seasonal, x='season', y='count', color='crime_type',
                          title='Crimes by Season and Type',
                          labels={'count': 'Number of Crimes', 'season': 'Season'})
        st.plotly_chart(fig_season, use_container_width=True)
    
    with tab2:
        st.markdown("### Geographic Crime Analysis")
        
        # City-wise distribution
        st.subheader("Crime Distribution by City")
        city_crimes = df_filtered['city'].value_counts().reset_index()
        city_crimes.columns = ['city', 'count']
        
        fig_city = px.bar(city_crimes, x='city', y='count',
                         title='Total Crimes by City',
                         labels={'count': 'Number of Crimes', 'city': 'City'})
        st.plotly_chart(fig_city, use_container_width=True)
        
        # Top neighborhoods
        st.subheader("Top 10 Neighborhoods by Crime Rate")
        hood_crimes = df_filtered['neighborhood'].value_counts().head(10)
        fig_hood = px.bar(x=hood_crimes.index, y=hood_crimes.values,
                         title='Most Dangerous Neighborhoods',
                         labels={'x': 'Neighborhood', 'y': 'Number of Crimes'})
        st.plotly_chart(fig_hood, use_container_width=True)
        
        # Crime density by city and type
        st.subheader("Crime Type Distribution by City")
        city_crime_matrix = pd.crosstab(df_filtered['city'], df_filtered['crime_type'])
        fig_heatmap = px.imshow(city_crime_matrix,
                               title='Crime Types Across Cities',
                               labels=dict(x="Crime Type", y="City", color="Count"))
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        st.markdown("### Crime Pattern Analysis")
        
        # Crime type distribution
        st.subheader("Crime Type Distribution")
        crime_dist = df_filtered['crime_type'].value_counts()
        fig_pie = px.pie(values=crime_dist.values, names=crime_dist.index,
                        title='Proportion of Crime Types')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Severity analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Crime Severity Distribution")
            severity_dist = df_filtered['severity'].value_counts().sort_index()
            fig_sev = px.bar(x=severity_dist.index, y=severity_dist.values,
                           labels={'x': 'Severity Level', 'y': 'Count'},
                           title='Distribution by Severity')
            st.plotly_chart(fig_sev, use_container_width=True)
        
        with col2:
            st.subheader("Average Severity by Crime Type")
            avg_sev = df_filtered.groupby('crime_type')['severity'].mean().sort_values(ascending=False)
            fig_avg_sev = px.bar(x=avg_sev.values, y=avg_sev.index, orientation='h',
                               labels={'x': 'Average Severity', 'y': 'Crime Type'},
                               title='Average Severity by Crime Type')
            st.plotly_chart(fig_avg_sev, use_container_width=True)
        
        # Arrest analysis
        st.subheader("Arrest Rate Analysis")
        arrest_by_crime = df_filtered.groupby('crime_type')['arrested'].mean() * 100
        arrest_by_crime = arrest_by_crime.sort_values(ascending=False)
        
        fig_arrest = px.bar(x=arrest_by_crime.values, y=arrest_by_crime.index, orientation='h',
                          labels={'x': 'Arrest Rate (%)', 'y': 'Crime Type'},
                          title='Arrest Rate by Crime Type')
        st.plotly_chart(fig_arrest, use_container_width=True)
    
    with tab4:
        st.markdown("### Demographic Analysis")
        
        # Victim age distribution
        st.subheader("Victim Age Distribution")
        valid_ages = df_filtered[df_filtered['victim_age'].notna()]
        
        fig_age = px.histogram(valid_ages, x='victim_age', nbins=30,
                              title='Distribution of Victim Ages',
                              labels={'victim_age': 'Age', 'count': 'Number of Victims'})
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Victim gender distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Victim Gender Distribution")
            gender_dist = df_filtered['victim_gender'].value_counts()
            fig_gender = px.pie(values=gender_dist.values, names=gender_dist.index,
                              title='Victims by Gender')
            st.plotly_chart(fig_gender, use_container_width=True)
        
        with col2:
            st.subheader("Domestic Violence Cases")
            dv_count = df_filtered['domestic_violence'].value_counts()
            fig_dv = px.pie(values=dv_count.values, names=['Non-DV', 'Domestic Violence'],
                          title='Domestic Violence Proportion')
            st.plotly_chart(fig_dv, use_container_width=True)
        
        # Age by crime type
        st.subheader("Age Distribution by Crime Type")
        fig_age_crime = px.box(valid_ages, x='crime_type', y='victim_age',
                              title='Victim Age Distribution by Crime Type',
                              labels={'crime_type': 'Crime Type', 'victim_age': 'Age'})
        fig_age_crime.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_age_crime, use_container_width=True)
    
    with tab5:
        st.markdown("### Summary Report")
        
        # Overall statistics
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Information**")
            st.write(f"- **Total Records:** {len(df_filtered):,}")
            st.write(f"- **Date Range:** {df_filtered['date'].min().date()} to {df_filtered['date'].max().date()}")
            st.write(f"- **Number of Cities:** {df_filtered['city'].nunique()}")
            st.write(f"- **Number of Crime Types:** {df_filtered['crime_type'].nunique()}")
            st.write(f"- **Missing Values:** {df_filtered.isnull().sum().sum()}")
        
        with col2:
            st.markdown("**Key Rates**")
            arrest_rate = (df_filtered['arrested'].sum() / len(df_filtered) * 100)
            dv_rate = (df_filtered['domestic_violence'].sum() / len(df_filtered) * 100)
            clearance_rate = (df_filtered[df_filtered['case_status'] == 'Closed'].shape[0] / len(df_filtered) * 100)
            
            st.write(f"- **Arrest Rate:** {arrest_rate:.2f}%")
            st.write(f"- **Domestic Violence Rate:** {dv_rate:.2f}%")
            st.write(f"- **Case Clearance Rate:** {clearance_rate:.2f}%")
        
        # Top statistics
        st.subheader("Top Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Most Common Crime Types**")
            top_crimes = df_filtered['crime_type'].value_counts().head(5)
            for crime, count in top_crimes.items():
                st.write(f"- {crime}: {count} ({count/len(df_filtered)*100:.1f}%)")
        
        with col2:
            st.markdown("**Cities with Highest Crime Rates**")
            top_cities = df_filtered['city'].value_counts().head(5)
            for city, count in top_cities.items():
                st.write(f"- {city}: {count} ({count/len(df_filtered)*100:.1f}%)")
        
        # Time patterns
        st.subheader("Peak Crime Times")
        col1, col2 = st.columns(2)
        
        with col1:
            peak_hour = df_filtered['hour'].mode()[0]
            st.write(f"**Peak Hour:** {peak_hour}:00 - {peak_hour+1}:00")
        
        with col2:
            peak_day = df_filtered['day_of_week'].mode()[0]
            st.write(f"**Peak Day:** {peak_day}")
        
        # Download filtered data
        st.subheader("Download Data")
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"crime_data_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    st.warning("Please run generate_data.py first to create the crime dataset.")

# Footer
st.markdown("---")
st.markdown("🚔 Crime Rate EDA Dashboard | Created with Streamlit, Python, and Love for Data Science")