# 📊 Amazon Sales Analysis & Sentiment Analysis Dashboard

A comprehensive Streamlit application for analyzing Amazon sales data with advanced sentiment analysis and predictive modeling. 

## 🌟 Features

### 📊 Data Overview
- Upload and analyze Amazon sales CSV data
- View statistical summaries and missing value analysis
- Explore complete dataset with customizable row display

### 🎨 Visualizations
- **Profit vs Discount**:  Analyze profit trends with discount percentages
- **Sales vs Discount**: Visualize sales performance across discounts
- **Rating vs Discount**: Explore customer satisfaction patterns
- **Price Analysis**: Examine price differences and distributions
- **Category Analysis**: Top 10 product categories breakdown

### 💭 Sentiment Analysis
- Automatic sentiment scoring using VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Sentiment classification:  Positive, Neutral, Negative
- Category-wise sentiment analysis
- Sentiment vs discount relationship visualization

### 🤖 Predictive Model
- Linear Regression model for profit prediction
- Discount impact analysis
- Real-time profit prediction based on discount percentage
- Model performance metrics (R², MAE, RMSE)

### 📥 Export Options
- Download processed dataset
- Export sentiment analysis results
- Complete data export with all analyses

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/amazon-sales-analysis. git
cd amazon-sales-analysis
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
amazon-sales-analysis/
├── app.py                  # Main Streamlit application
├── requirements. txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── data/
│   └── sample_data.csv    # Sample Amazon sales data
└── utils/
    └── data_processor.py  # Data processing utilities (optional)
```

## 📋 Expected CSV Format

Your Amazon sales CSV should contain these columns: 

| Column | Type | Description |
|--------|------|-------------|
| `product_name` | String | Name of the product |
| `discounted_price` | Float | Selling/discounted price |
| `actual_price` | Float | Original/actual price |
| `discount_percentage` | Float | Discount percentage |
| `rating` | Float | Product rating (0-5) |
| `rating_count` | Integer | Number of ratings |
| `about_product` | String | Product description |
| `category` | String | Product category (pipe-separated) |
| `Quantity` | Integer | Units sold |

## 🔧 Usage

### 1. Upload Data
- Click on "Upload your Amazon CSV file" in the sidebar
- Select your CSV file containing Amazon sales data

### 2. Explore Tabs
- **Data Overview**: View and analyze raw data statistics
- **Visualizations**:  Explore visual representations of sales metrics
- **Sentiment Analysis**:  Analyze product descriptions sentiment
- **Predictive Model**:  Make profit predictions based on discounts
- **Download Results**: Export analysis results

### 3. Download Results
- Select the data type you want to export
- Click the download button to save the results

## 📊 Data Processing Pipeline

1. **Data Cleaning**
   - Remove missing values in critical columns
   - Convert data types appropriately
   - Handle currency symbols and formatting

2. **Feature Engineering**
   - Calculate Sales:  `discounted_price × Quantity`
   - Calculate Cost: `(actual_price × 0.7) × Quantity`
   - Calculate Profit: `Sales - Cost`
   - Calculate Price Difference: `actual_price - discounted_price`

3. **Category Splitting**
   - Split pipe-separated categories into main, sub1, and sub2

4. **Sentiment Analysis**
   - Analyze product descriptions using VADER
   - Generate sentiment scores (-1 to 1)
   - Classify into Positive, Neutral, Negative

## 🤖 Predictive Model Details

### Algorithm:  Linear Regression

- **Target Variable**: Profit
- **Feature**:  Discount Percentage
- **Train-Test Split**: 80-20
- **Evaluation Metrics**:
  - R² Score:  Model fit quality
  - MAE: Mean Absolute Error
  - RMSE: Root Mean Squared Error

### Interpretation
- **Positive Coefficient**: Higher discounts increase profit (demand-driven)
- **Negative Coefficient**: Higher discounts decrease profit (margin-driven)

## 🎨 Customization

### Change Color Scheme
Edit the `st.markdown()` styling section in `app.py`:

```python
st.markdown("""
<style>
    .header { color: #FF9900; }  # Change header color
    .subheader { color: #146EB4; }  # Change subheader color
</style>
""", unsafe_allow_html=True)
```

### Adjust Analysis Parameters
- Sentiment thresholds (default: -0.05 to 0.05)
- Cost ratio (default: 70% of actual price)
- Model test size (default: 20%)

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. Push your repository to GitHub
2. Sign up at [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and select your repository
4. Streamlit will automatically deploy your app

### Deploy to Heroku

1. Add `Procfile`:
```
web: streamlit run app.py --logger.level=error
```

2. Deploy:
```bash
heroku login
heroku create your-app-name
git push heroku main
```

### Deploy to AWS/Azure/GCP

Use containerization with Docker.  See Streamlit deployment documentation.

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Ensure all dependencies are installed: 
```bash
pip install -r requirements.txt
```

### Issue: "NLTK data not found"
**Solution**: The app automatically downloads required NLTK data on first run.  If it fails: 
```python
import nltk
nltk.download('vader_lexicon')
```

### Issue: File upload not working
**Solution**: Ensure CSV format is correct and uses UTF-8 encoding

## 📈 Performance Tips

- Use data with <100k rows for optimal performance
- CSV files should be <50MB
- Clear browser cache if experiencing slow loads

## 🤝 Contributing

Contributions are welcome! Please: 

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Daniel Usuwa**
- GitHub: [@danielusuwa-collab](https://github.com/danielusuwa-collab)
- LinkedIn: [Your LinkedIn Profile]

## 🙏 Acknowledgments

- VADER Sentiment Analysis (NLTK)
- Streamlit for the amazing framework
- Pandas, NumPy, Scikit-learn communities

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email:  your-email@example.com
- LinkedIn: [Your Profile]

---

**Last Updated**: December 12, 2025
**Version**: 1.0.0