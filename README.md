# Hotel Review Sentiment Analyzer

A comprehensive sentiment analysis system for hotel reviews using Natural Language Processing techniques with Python and TextBlob.

## 🎯 Project Overview

This project implements a complete sentiment analysis pipeline for hotel reviews, capable of:
- Classifying reviews into Positive, Negative, and Neutral sentiments
- Extracting service-related topics (Room Service, Food Quality, Staff Service, etc.)
- Generating comprehensive business intelligence insights
- Creating professional data visualizations
- Exporting detailed analysis results

## 🛠️ Technologies Used

- **Python 3.8+** - Core programming language
- **TextBlob** - Natural Language Processing and sentiment analysis
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization

## 📊 Features

### Sentiment Analysis
- **Polarity Scoring**: Ranges from -1 (most negative) to +1 (most positive)
- **Subjectivity Analysis**: Measures objectivity vs subjectivity (0 to 1)
- **Enhanced Classification**: Uses ±0.2 thresholds for improved accuracy
- **Confidence Scoring**: Provides reliability metrics for classifications

### Service Topic Extraction
- **6 Service Categories**: Room Service, Food Quality, Staff Service, Facilities, Location, Cleanliness
- **Keyword-based Detection**: Uses comprehensive domain-specific dictionaries
- **Relevance Scoring**: Ranks topics by mention frequency and context
- **Multi-label Classification**: Reviews can be tagged with multiple service categories

### Data Visualization
- **Sentiment Distribution Charts**: Pie charts and bar graphs
- **Service Topic Analysis**: Frequency and quality scoring visualizations
- **Trend Analysis**: Temporal sentiment patterns
- **Business Intelligence Dashboards**: Professional charts for reporting

### Export Capabilities
- **CSV Reports**: Detailed analysis results with all metrics
- **High-Resolution Charts**: PNG exports suitable for presentations
- **Business Insights**: Actionable recommendations for hotel management

## 🚀 Quick Start

### Installation

1. Clone this repository:
\`\`\`bash
git clone https://github.com/yourusername/hotel-review-sentiment-analyzer.git
cd hotel-review-sentiment-analyzer
\`\`\`

2. Install required dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. Download TextBlob corpora (first time only):
```python
import nltk
nltk.download('punkt')
nltk.download('brown')
