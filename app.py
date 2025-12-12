# -*- coding: utf-8 -*-
"""
Amazon Sales Analysis & Sentiment Analysis Dashboard
A comprehensive Streamlit application for Amazon sales data visualization and sentiment analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import nltk
from nltk.sentiment. vader import SentimentIntensityAnalyzer
import warnings

warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Amazon Sales Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== DOWNLOAD REQUIRED NLTK DATA ====================
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('vader_lexicon', quiet=True)
    except:
        st.warning("Could not download NLTK data. Some features may not work.")

download_nltk_data()

# ==================== CUSTOM STYLING ====================
st.markdown("""
<style>
    .header {
        text-align: center;
        color: #FF9900;
        font-size: 2.5em;
        margin-bottom: 1em;
    }
    .subheader {
        color: #146EB4;
        font-size: 1.5em;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius:  10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING & CACHING ====================
@st.cache_data
def load_data(uploaded_file):
    """Load and cache data from uploaded CSV"""
    df = pd.read_csv(uploaded_file)
    return df

@st.cache_data
def preprocess_data(df):
    """Clean and preprocess the dataset"""
    # Remove missing values in rating_count
    df = df.dropna(subset=['rating_count'])
    
    # Convert data types
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').replace('nan', '0').astype(int)
    df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').replace('nan', '0.0').astype(float)
    df['actual_price'] = df['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '').replace('nan', '0.0').astype(float)
    
    # Calculate new columns
    if 'Quantity' in df.columns:
        df['Sales'] = df['discounted_price'] * df['Quantity']
        df['Cost'] = (df['actual_price'] * 0.7) * df['Quantity']
        df['Profit'] = df['Sales'] - df['Cost']
        df[['Sales', 'Cost', 'Profit']] = df[['Sales', 'Cost', 'Profit']].round(2)
    
    # Price difference
    df["price_difference"] = df["actual_price"] - df["discounted_price"]
    
    # Split category column
    if 'category' in df.columns:
        categories = df['category'].str.split('|', expand=True)
        max_cols = min(3, categories.shape[1])
        col_names = ['category_main', 'category_sub1', 'category_sub2'][: max_cols]
        df[col_names] = categories. iloc[:, : max_cols]
        df. drop(columns=['category'], inplace=True)
    
    return df

@st.cache_data
def sentiment_analysis(df):
    """Perform sentiment analysis on product descriptions"""
    sid = SentimentIntensityAnalyzer()
    
    if 'about_product' not in df.columns:
        return df
    
    df['about_product'] = df['about_product'].astype(str)
    df['sentiment_score'] = df['about_product'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral')
    )
    
    return df

# ==================== MAIN APPLICATION ====================
def main():
    # Header
    st.markdown('<div class="header">📊 Amazon Sales Analysis Dashboard</div>', unsafe_allow_html=True)
    st.write("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your Amazon CSV file",
        type="csv",
        help="Select a CSV file containing Amazon sales data"
    )
    
    if uploaded_file is None:
        st.info("👉 Please upload a CSV file to get started!")
        st.markdown("""
        ### 📋 Expected Columns:
        - `product_name`: Name of the product
        - `discounted_price`: Selling price
        - `actual_price`: Original price
        - `discount_percentage`: Discount percentage
        - `rating`: Product rating
        - `rating_count`: Number of ratings
        - `about_product`: Product description
        - `category`: Product category
        - `Quantity`: Units sold (for profit calculation)
        """)
        return
    
    # Load and process data
    df = load_data(uploaded_file)
    df = preprocess_data(df)
    df = sentiment_analysis(df)
    
    # Display data info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Dataset Info")
    st.sidebar.metric("Total Records", len(df))
    st.sidebar.metric("Total Columns", len(df.columns))
    st.sidebar.metric("Missing Values", df.isnull().sum().sum())
    
    # Navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview",
        "🎨 Visualizations",
        "💭 Sentiment Analysis",
        "🤖 Predictive Model",
        "📥 Download Results"
    ])
    
    # ==================== TAB 1: DATA OVERVIEW ====================
    with tab1:
        st.markdown('<div class="subheader">Data Preview</div>', unsafe_allow_html=True)
        
        col1, col2 = st. columns([3, 1])
        with col1:
            num_rows = st.slider("Rows to display:", min_value=5, max_value=min(100, len(df)), value=10, step=5)
        with col2:
            st. write("")  # Spacer
        
        st.dataframe(df.head(num_rows), use_container_width=True)
        
        # Data statistics
        st.markdown("### 📊 Statistical Summary")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            st.dataframe(df[numeric_cols]. describe().round(2), use_container_width=True)
        
        # Missing values
        st.markdown("### 🔍 Missing Values")
        missing_data = df.isnull().sum()
        if missing_data. sum() > 0:
            st.dataframe(missing_data[missing_data > 0], use_container_width=True)
        else:
            st.success("✅ No missing values found!")
    
    # ==================== TAB 2: VISUALIZATIONS ====================
    with tab2:
        st.markdown('<div class="subheader">Data Visualizations</div>', unsafe_allow_html=True)
        
        viz_type = st.selectbox(
            "Select visualization type:",
            ["Profit vs Discount", "Sales vs Discount", "Rating vs Discount", "Price Analysis", "Category Analysis"]
        )
        
        if viz_type == "Profit vs Discount" and 'Profit' in df.columns:
            col1, col2 = st. columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(df['discount_percentage'], df['Profit'], color='blue', alpha=0.6, s=50)
                ax. set_title('Profit vs Discount Percentage', fontsize=14, fontweight='bold')
                ax.set_xlabel('Discount Percentage (%)')
                ax.set_ylabel('Profit (₹)')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            with col2:
                st.metric("Average Profit", f"₹{df['Profit'].mean():.2f}")
                st. metric("Max Profit", f"₹{df['Profit'].max():.2f}")
                st.metric("Min Profit", f"₹{df['Profit'].min():.2f}")
        
        elif viz_type == "Sales vs Discount" and 'Sales' in df.columns:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df['discount_percentage'], df['Sales'], color='green', marker='o', alpha=0.7)
                ax.set_title('Sales vs Discount Percentage', fontsize=14, fontweight='bold')
                ax.set_xlabel('Discount Percentage (%)')
                ax.set_ylabel('Total Sales (₹)')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            with col2:
                st.metric("Total Sales", f"₹{df['Sales'].sum():.2f}")
                st.metric("Avg Sale Value", f"₹{df['Sales'].mean():.2f}")
        
        elif viz_type == "Rating vs Discount": 
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt. subplots(figsize=(10, 5))
                ax.plot(df['discount_percentage'], df['rating'], color='orange', marker='o', alpha=0.7)
                ax.set_title('Rating vs Discount Percentage', fontsize=14, fontweight='bold')
                ax.set_xlabel('Discount Percentage (%)')
                ax.set_ylabel('Average Rating')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            with col2:
                st.metric("Avg Rating", f"{df['rating'].mean():.2f}")
                st.metric("Max Rating", f"{df['rating'].max():.2f}")
        
        elif viz_type == "Price Analysis":
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt. subplots(figsize=(8, 5))
                ax.hist(df['price_difference'], bins=30, color='skyblue', edgecolor='black')
                ax.set_title('Price Difference Distribution', fontsize=12, fontweight='bold')
                ax.set_xlabel('Price Difference (₹)')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(df['actual_price'], df['discounted_price'], alpha=0.5, color='red')
                ax.plot([df['actual_price'].min(), df['actual_price'].max()], 
                        [df['actual_price'].min(), df['actual_price'].max()], 
                        'k--', lw=2)
                ax. set_title('Actual vs Discounted Price', fontsize=12, fontweight='bold')
                ax.set_xlabel('Actual Price (₹)')
                ax.set_ylabel('Discounted Price (₹)')
                st.pyplot(fig)
        
        elif viz_type == "Category Analysis" and 'category_main' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 5))
            category_counts = df['category_main'].value_counts().head(10)
            category_counts.plot(kind='barh', ax=ax, color='teal')
            ax.set_title('Top 10 Product Categories', fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Products')
            st.pyplot(fig)
    
    # ==================== TAB 3: SENTIMENT ANALYSIS ====================
    with tab3:
        st.markdown('<div class="subheader">Sentiment Analysis</div>', unsafe_allow_html=True)
        
        if 'sentiment_label' in df.columns:
            col1, col2, col3 = st.columns(3)
            
            positive_count = len(df[df['sentiment_label'] == 'Positive'])
            neutral_count = len(df[df['sentiment_label'] == 'Neutral'])
            negative_count = len(df[df['sentiment_label'] == 'Negative'])
            
            with col1:
                st.metric("😊 Positive", positive_count, f"{positive_count/len(df)*100:.1f}%")
            with col2:
                st.metric("😐 Neutral", neutral_count, f"{neutral_count/len(df)*100:.1f}%")
            with col3:
                st.metric("😞 Negative", negative_count, f"{negative_count/len(df)*100:.1f}%")
            
            st.write("---")
            
            col1, col2 = st.columns(2)
            
            # Sentiment Distribution
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                sentiment_counts = df['sentiment_label'].value_counts()
                colors = {'Positive': '#2ecc71', 'Neutral':  '#95a5a6', 'Negative': '#e74c3c'}
                sentiment_counts.plot(kind='bar', ax=ax, color=[colors. get(x, 'grey') for x in sentiment_counts.index])
                ax.set_title('Sentiment Distribution', fontsize=12, fontweight='bold')
                ax. set_xlabel('Sentiment')
                ax.set_ylabel('Count')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
                st.pyplot(fig)
            
            # Sentiment Pie Chart
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                colors_pie = ['#2ecc71', '#95a5a6', '#e74c3c']
                ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                       colors=colors_pie, startangle=90)
                ax.set_title('Sentiment Breakdown', fontsize=12, fontweight='bold')
                st. pyplot(fig)
            
            st.write("---")
            
            # Sentiment by Category
            if 'category_main' in df.columns:
                st.markdown("### Sentiment by Category")
                sent_by_cat = df.groupby('category_main')['sentiment_score'].mean().sort_values(ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sent_by_cat.plot(kind='barh', ax=ax, color='teal')
                ax.set_title('Average Sentiment Score by Category (Top 10)', fontsize=12, fontweight='bold')
                ax.set_xlabel('Sentiment Score')
                st.pyplot(fig)
            
            # Sentiment vs Discount
            st.markdown("### Sentiment vs Discount Percentage")
            fig, ax = plt.subplots(figsize=(10, 5))
            for sentiment in df['sentiment_label'].unique():
                mask = df['sentiment_label'] == sentiment
                ax.scatter(df[mask]['discount_percentage'], df[mask]['sentiment_score'], 
                          label=sentiment, alpha=0.6, s=50)
            ax.set_title('Sentiment Score vs Discount Percentage', fontsize=12, fontweight='bold')
            ax.set_xlabel('Discount Percentage (%)')
            ax.set_ylabel('Sentiment Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning("Sentiment analysis data not available. Please ensure 'about_product' column exists.")
    
    # ==================== TAB 4: PREDICTIVE MODEL ====================
    with tab4:
        st.markdown('<div class="subheader">Predictive Model - Discount Impact Analysis</div>', unsafe_allow_html=True)
        
        if 'Profit' in df.columns:
            st.info("Building a Linear Regression model to predict profit based on discount percentage...")
            
            # Prepare data
            X = df[['discount_percentage']].values
            y = df['Profit'].values
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{r2:. 4f}")
            with col2:
                st.metric("MAE", f"₹{mae:.2f}")
            with col3:
                st.metric("RMSE", f"₹{rmse:. 2f}")
            
            st.write("---")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual', s=50)
            ax.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
            ax.set_title('Actual vs Predicted Profit', fontsize=12, fontweight='bold')
            ax.set_xlabel('Discount Percentage (%)')
            ax.set_ylabel('Profit (₹)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.write("---")
            
            # Prediction interface
            st.markdown("### 🔮 Make Predictions")
            discount_input = st.slider(
                "Enter discount percentage:",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0
            )
            
            predicted_profit = model.predict([[discount_input]])[0]
            
            col1, col2 = st. columns(2)
            with col1:
                st.metric("Discount Percentage", f"{discount_input:.1f}%")
            with col2:
                if predicted_profit >= 0:
                    st. metric("Predicted Profit", f"₹{predicted_profit:. 2f}", delta="Positive ✅")
                else:
                    st.metric("Predicted Profit", f"₹{predicted_profit:.2f}", delta="Negative ⚠️")
            
            # Model insights
            st.markdown("### 📊 Model Insights")
            st.write(f"**Coefficient (Slope):** {model.coef_[0]:.4f}")
            st.write(f"**Intercept:** {model.intercept_:.2f}")
            st.markdown("""
            - **Coefficient**: Shows how profit changes with each 1% increase in discount
            - **Positive coefficient**: Higher discounts → Higher profit (demand-driven)
            - **Negative coefficient**: Higher discounts → Lower profit (margin-driven)
            """)
        else:
            st.warning("Profit data not available. Please ensure data preprocessing includes Quantity column.")
    
    # ==================== TAB 5: DOWNLOAD RESULTS ====================
    with tab5:
        st.markdown('<div class="subheader">Export Results</div>', unsafe_allow_html=True)
        
        export_option = st.radio(
            "Select what to export:",
            ["Processed Dataset", "Sentiment Analysis Results", "All Data"]
        )
        
        if export_option == "Processed Dataset":
            csv = df.drop(columns=['sentiment_score', 'sentiment_label'], errors='ignore').to_csv(index=False)
            st.download_button(
                label="📥 Download Processed Data",
                data=csv,
                file_name="Cleaned_AmazonSales. csv",
                mime="text/csv"
            )
        
        elif export_option == "Sentiment Analysis Results":
            if 'sentiment_label' in df.columns:
                sentiment_df = df[['product_name', 'about_product', 'sentiment_score', 'sentiment_label']]. copy()
                csv = sentiment_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Sentiment Analysis",
                    data=csv,
                    file_name="Sentiment_Analysis_Results.csv",
                    mime="text/csv"
                )
        
        elif export_option == "All Data": 
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete Dataset",
                data=csv,
                file_name="Complete_Analysis_Data.csv",
                mime="text/csv"
            )
        
        st.write("---")
        st.markdown("### 📋 Dataset Summary")
        st.write(f"**Total Records:** {len(df)}")
        st.write(f"**Total Columns:** {len(df. columns)}")
        st.write(f"**File Size:** ~{len(df) * len(df.columns) * 8 / 1024 / 1024:.2f} MB")

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()