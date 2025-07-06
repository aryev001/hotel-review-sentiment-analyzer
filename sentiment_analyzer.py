"""
Hotel Review Sentiment Analyzer
A comprehensive sentiment analysis system for hotel reviews using TextBlob and Python.
Author: [Your Name]
Date: January 2024
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HotelReviewSentimentAnalyzer:
    def __init__(self):
        self.reviews_data = None
        self.analyzed_data = None
        self.service_keywords = {
            'room_service': ['room service', 'housekeeping', 'cleaning', 'towels', 'bed', 'room', 'bedroom'],
            'food_quality': ['food', 'restaurant', 'breakfast', 'dinner', 'meal', 'kitchen', 'dining', 'buffet'],
            'staff_service': ['staff', 'reception', 'front desk', 'employee', 'service', 'helpful', 'friendly'],
            'facilities': ['pool', 'gym', 'spa', 'wifi', 'parking', 'elevator', 'amenities', 'fitness'],
            'location': ['location', 'nearby', 'transport', 'airport', 'downtown', 'beach', 'city center'],
            'cleanliness': ['clean', 'dirty', 'hygiene', 'sanitized', 'tidy', 'mess', 'spotless']
        }
        
    def load_data(self, csv_file_path):
        """Load hotel review data from CSV file"""
        try:
            self.reviews_data = pd.read_csv(csv_file_path)
            print(f"✅ Loaded {len(self.reviews_data)} reviews from {csv_file_path}")
            return self.reviews_data
        except FileNotFoundError:
            print(f"❌ File {csv_file_path} not found. Using sample data instead.")
            return self.create_sample_dataset()
    
    def create_sample_dataset(self):
        """Create a sample dataset if CSV file is not available"""
        
        hotel_reviews = [
            # Positive Reviews
            "The hotel exceeded all my expectations! The room was spacious and immaculately clean. The housekeeping staff did an outstanding job maintaining everything. The front desk staff were incredibly helpful and friendly throughout our stay. The breakfast buffet had a wonderful variety of fresh options. The location is perfect for exploring the city center. The pool area was beautiful and well-maintained. Highly recommend this hotel to anyone visiting!",
            
            "Absolutely fantastic experience! The room service was prompt and the food quality was exceptional. Our room had a stunning view and was equipped with all modern amenities. The staff went above and beyond to make our anniversary special. The spa services were incredibly relaxing and professional. The hotel's location made it easy to walk to restaurants and attractions. Will definitely be returning!",
            
            "Perfect business hotel! The wifi was fast and reliable throughout the property. The room had an excellent workspace setup. The front desk staff were professional and efficient with check-in and check-out. Great location near the business district with convenient airport access. The gym facilities are modern and well-equipped. The executive lounge was a nice touch with complimentary refreshments.",
            
            "Outstanding family-friendly hotel! The kids absolutely loved the pool area and the staff provided extra towels without us even asking. The rooms are spacious enough for our family of four. The restaurant staff were amazing at accommodating our dietary restrictions during breakfast. The hotel feels very safe and secure. Excellent value for money and we'll definitely return for our next family vacation!",
            
            "Luxury at its finest! The room was elegantly decorated with high-end furnishings. The housekeeping attention to detail was impeccable. The concierge helped us plan our entire itinerary with excellent recommendations. The hotel restaurant serves gourmet food that rivals the best restaurants in the city. The spa treatments were world-class. Every staff member we encountered was professional and courteous.",
            
            # Negative Reviews
            "Extremely disappointing experience. Our room was not ready at the scheduled check-in time and we had to wait over 2 hours in the lobby. When we finally got to our room, it was dirty with hair in the bathroom and visible stains on the carpet. The air conditioning was completely broken and it took them an entire day to send someone to fix it. The front desk staff seemed overwhelmed and were not helpful at all.",
            
            "Terrible value for the price we paid. The room was incredibly small and felt cramped. The furniture looked old and worn out. The wifi was extremely slow and kept disconnecting every few minutes. The breakfast was awful - cold food with very limited options. The elevator was out of order for our entire 3-day stay, forcing us to use the stairs to reach the 8th floor. The parking fees were absolutely outrageous.",
            
            "Worst customer service I've ever experienced at a hotel. The reception staff were rude and unprofessional from the moment we arrived. Our room wasn't cleaned properly - we found dirty towels still hanging in the bathroom and the bed wasn't even made. We complained multiple times but nothing was done to address our concerns. The restaurant food was cold and overpriced. The pool was closed for maintenance without any prior notice to guests.",
            
            "Absolutely unacceptable conditions. The room had a strong cigarette smell despite being advertised as non-smoking. The bathroom had visible mold around the shower area and there was no hot water for the entire duration of our stay. The noise from the street was unbearable - we couldn't sleep at all during the night. The housekeeping staff entered our room without permission while we were out. I would never recommend this hotel to anyone.",
            
            "Complete waste of money. The hotel location is terrible with no restaurants or attractions within walking distance. The room service took over an hour to deliver cold food. The gym equipment is old, broken, and clearly hasn't been maintained. The staff don't speak English well which made communication extremely difficult. The wifi doesn't work in the rooms, only in the lobby. Very frustrating and disappointing stay.",
            
            # Neutral Reviews
            "Average hotel experience overall. The room was clean and adequately sized but nothing particularly impressive. The staff were polite and professional but not exceptionally helpful. The breakfast offered standard continental options - nothing special but acceptable. The location is decent with some restaurants and shops within walking distance. The wifi worked reliably throughout our stay. It's an acceptable choice if you just need a place to sleep.",
            
            "Mixed feelings about this hotel stay. The room was comfortable with a good bed, but the bathroom was quite small and cramped. The front desk staff were friendly during check-in but seemed very busy and rushed during the rest of our stay. The hotel restaurant food was decent quality but significantly overpriced for what you get. The pool area was nice but always crowded. Overall, it was okay but not exceptional.",
            
            "Standard business hotel that meets basic expectations. The rooms are clean and functional with reliable amenities. The gym is small but has the essential equipment needed for a workout. The breakfast is typical continental style with the usual options you'd expect. The location is convenient for business meetings with good access to public transportation. The wifi is stable and fast. Nothing exceptional but it serves its purpose for a short business trip.",
            
            "Decent hotel for the price point. The room was clean and the bed was comfortable enough for a good night's sleep. The housekeeping staff did a thorough job cleaning our room daily. The staff were professional throughout our stay but not overly friendly or engaging. The hotel amenities are basic but functional. The location offers good access to public transport. The parking is available but comes with an additional fee. Would consider staying again if the price is right.",
            
            "Okay experience that met our basic needs. The check-in process was smooth and efficient, and our room was ready exactly on time. The housekeeping staff maintained the room well throughout our stay. The hotel restaurant offers average food with a standard menu - nothing exciting but not bad either. The facilities are well-maintained but not luxurious or modern. Good value for budget-conscious travelers who don't need luxury amenities."
        ]
        
        # Create DataFrame with additional metadata
        np.random.seed(42)  # For reproducible results
        
        self.reviews_data = pd.DataFrame({
            'review_id': range(1, len(hotel_reviews) + 1),
            'review_text': hotel_reviews,
            'hotel_name': np.random.choice(['Grand Plaza Hotel', 'City Center Inn', 'Business Suites', 'Family Resort', 'Luxury Palace'], len(hotel_reviews)),
            'reviewer_location': np.random.choice(['New York', 'London', 'Tokyo', 'Paris', 'Sydney', 'Toronto', 'Berlin'], len(hotel_reviews)),
            'stay_duration': np.random.choice([1, 2, 3, 4, 5, 7], len(hotel_reviews)),
            'travel_type': np.random.choice(['Business', 'Leisure', 'Family', 'Couple', 'Solo'], len(hotel_reviews)),
            'date_posted': pd.date_range(start='2023-01-01', periods=len(hotel_reviews), freq='W')
        })
        
        print(f"✅ Created sample dataset with {len(self.reviews_data)} hotel reviews")
        return self.reviews_data
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob with enhanced accuracy"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Enhanced sentiment classification
        if polarity > 0.2:
            return 'Positive', polarity, subjectivity
        elif polarity < -0.2:
            return 'Negative', polarity, subjectivity
        else:
            return 'Neutral', polarity, subjectivity
    
    def extract_service_topics(self, text):
        """Extract service-related topics from review text"""
        text_lower = text.lower()
        found_topics = []
        topic_scores = {}
        
        for topic, keywords in self.service_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += text_lower.count(keyword)
            
            if score > 0:
                found_topics.append(topic)
                topic_scores[topic] = score
        
        return found_topics if found_topics else ['general'], topic_scores
    
    def process_all_reviews(self):
        """Process all reviews for comprehensive analysis"""
        if self.reviews_data is None:
            print("❌ No data available. Please load data first.")
            return
        
        print("🔄 Processing reviews for sentiment analysis...")
        
        # Initialize lists for results
        sentiments = []
        polarities = []
        subjectivities = []
        service_topics = []
        topic_scores = []
        
        # Process each review
        for idx, review in enumerate(self.reviews_data['review_text']):
            # Sentiment analysis
            sentiment, polarity, subjectivity = self.analyze_sentiment(review)
            sentiments.append(sentiment)
            polarities.append(polarity)
            subjectivities.append(subjectivity)
            
            # Topic extraction
            topics, scores = self.extract_service_topics(review)
            service_topics.append(topics)
            topic_scores.append(scores)
            
            # Progress indicator
            if (idx + 1) % 5 == 0:
                print(f"   Processed {idx + 1}/{len(self.reviews_data)} reviews...")
        
        # Add results to dataframe
        self.reviews_data['sentiment'] = sentiments
        self.reviews_data['polarity_score'] = polarities
        self.reviews_data['subjectivity_score'] = subjectivities
        self.reviews_data['service_topics'] = service_topics
        self.reviews_data['topic_scores'] = topic_scores
        
        self.analyzed_data = self.reviews_data.copy()
        print("✅ Analysis completed successfully!")
        
        return self.analyzed_data
    
    def generate_insights(self):
        """Generate comprehensive insights from analyzed data"""
        if self.analyzed_data is None:
            print("❌ No analyzed data available. Please process reviews first.")
            return None
        
        insights = {}
        
        # Overall sentiment statistics
        sentiment_counts = self.analyzed_data['sentiment'].value_counts()
        insights['sentiment_distribution'] = sentiment_counts.to_dict()
        insights['total_reviews'] = len(self.analyzed_data)
        insights['avg_polarity'] = self.analyzed_data['polarity_score'].mean()
        insights['avg_subjectivity'] = self.analyzed_data['subjectivity_score'].mean()
        
        # Service topic analysis
        all_topics = []
        for topics_list in self.analyzed_data['service_topics']:
            all_topics.extend(topics_list)
        
        topic_frequency = Counter(all_topics)
        insights['service_topics_frequency'] = dict(topic_frequency)
        
        # Sentiment by service topic
        topic_sentiment_analysis = {}
        for topic in self.service_keywords.keys():
            topic_reviews = self.analyzed_data[
                self.analyzed_data['service_topics'].apply(lambda x: topic in x)
            ]
            if len(topic_reviews) > 0:
                topic_sentiment_analysis[topic] = {
                    'total_mentions': len(topic_reviews),
                    'positive': len(topic_reviews[topic_reviews['sentiment'] == 'Positive']),
                    'negative': len(topic_reviews[topic_reviews['sentiment'] == 'Negative']),
                    'neutral': len(topic_reviews[topic_reviews['sentiment'] == 'Neutral']),
                    'avg_polarity': topic_reviews['polarity_score'].mean()
                }
        
        insights['topic_sentiment_analysis'] = topic_sentiment_analysis
        
        # Identify strengths and pain points
        positive_reviews = self.analyzed_data[self.analyzed_data['sentiment'] == 'Positive']
        negative_reviews = self.analyzed_data[self.analyzed_data['sentiment'] == 'Negative']
        
        positive_topics = []
        negative_topics = []
        
        for topics_list in positive_reviews['service_topics']:
            positive_topics.extend(topics_list)
        
        for topics_list in negative_reviews['service_topics']:
            negative_topics.extend(topics_list)
        
        insights['top_strengths'] = dict(Counter(positive_topics).most_common(5))
        insights['top_pain_points'] = dict(Counter(negative_topics).most_common(5))
        
        return insights
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if self.analyzed_data is None:
            print("❌ No analyzed data available.")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hotel Review Sentiment Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Distribution Pie Chart
        sentiment_counts = self.analyzed_data['sentiment'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        wedges, texts, autotexts = axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                                                 autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0, 0].set_title('Overall Sentiment Distribution', fontweight='bold')
        
        # 2. Polarity Score Distribution
        axes[0, 1].hist(self.analyzed_data['polarity_score'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Sentiment Polarity Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Polarity Score (-1 to +1)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Service Topics Frequency
        all_topics = []
        for topics_list in self.analyzed_data['service_topics']:
            all_topics.extend([topic for topic in topics_list if topic != 'general'])
        
        if all_topics:
            topic_counts = Counter(all_topics)
            topics = list(topic_counts.keys())
            counts = list(topic_counts.values())
            
            bars = axes[1, 0].bar(topics, counts, color='lightcoral', alpha=0.8)
            axes[1, 0].set_title('Service Topics Mentioned', fontweight='bold')
            axes[1, 0].set_xlabel('Service Topics')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{int(height)}', ha='center', va='bottom')
        
        # 4. Sentiment by Travel Type
        if 'travel_type' in self.analyzed_data.columns:
            travel_sentiment = self.analyzed_data.groupby('travel_type')['sentiment'].value_counts().unstack(fill_value=0)
            travel_sentiment.plot(kind='bar', stacked=True, ax=axes[1, 1], 
                                color=['#e74c3c', '#f39c12', '#2ecc71'])
            axes[1, 1].set_title('Sentiment Distribution by Travel Type', fontweight='bold')
            axes[1, 1].set_xlabel('Travel Type')
            axes[1, 1].set_ylabel('Number of Reviews')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].legend(title='Sentiment')
        
        plt.tight_layout()
        plt.savefig('sentiment_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizations created and saved as 'sentiment_analysis_results.png'")
    
    def save_results(self, filename='hotel_sentiment_analysis_results.csv'):
        """Save analysis results to CSV file"""
        if self.analyzed_data is None:
            print("❌ No analyzed data available.")
            return None
        
        # Prepare data for export
        export_data = self.analyzed_data.copy()
        
        # Convert service topics list to string
        export_data['service_topics_str'] = export_data['service_topics'].apply(lambda x: ', '.join(x))
        
        # Select columns for export
        columns_to_export = [
            'review_id', 'hotel_name', 'reviewer_location', 'travel_type', 
            'stay_duration', 'date_posted', 'review_text', 'sentiment', 
            'polarity_score', 'subjectivity_score', 'service_topics_str'
        ]
        
        # Only include columns that exist in the dataframe
        available_columns = [col for col in columns_to_export if col in export_data.columns]
        export_df = export_data[available_columns].copy()
        
        # Save to CSV
        export_df.to_csv(filename, index=False)
        print(f"✅ Results saved to '{filename}'")
        
        return export_df
    
    def print_summary(self):
        """Print comprehensive analysis summary"""
        insights = self.generate_insights()
        
        if insights is None:
            return
        
        print("\n" + "="*80)
        print("🏨 HOTEL REVIEW SENTIMENT ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   Total Reviews Analyzed: {insights['total_reviews']}")
        print(f"   Average Sentiment Score: {insights['avg_polarity']:.3f}")
        print(f"   Average Subjectivity: {insights['avg_subjectivity']:.3f}")
        
        print(f"\n📈 SENTIMENT DISTRIBUTION:")
        total = insights['total_reviews']
        for sentiment, count in insights['sentiment_distribution'].items():
            percentage = (count / total) * 100
            print(f"   {sentiment}: {count} reviews ({percentage:.1f}%)")
        
        print(f"\n🏨 SERVICE TOPICS ANALYSIS:")
        print("   Most Mentioned Service Categories:")
        for topic, count in sorted(insights['service_topics_frequency'].items(), 
                                 key=lambda x: x[1], reverse=True):
            if topic != 'general':
                print(f"   • {topic.replace('_', ' ').title()}: {count} mentions")
        
        print(f"\n✅ TOP SERVICE STRENGTHS:")
        for i, (topic, count) in enumerate(insights['top_strengths'].items(), 1):
            if topic != 'general':
                print(f"   {i}. {topic.replace('_', ' ').title()}: {count} positive mentions")
        
        print(f"\n❌ TOP SERVICE PAIN POINTS:")
        for i, (topic, count) in enumerate(insights['top_pain_points'].items(), 1):
            if topic != 'general':
                print(f"   {i}. {topic.replace('_', ' ').title()}: {count} negative mentions")
        
        print("\n" + "="*80)

def main():
    """Main function to run the hotel review sentiment analyzer"""
    print("🏨 HOTEL REVIEW SENTIMENT ANALYZER")
    print("=" * 60)
    print("📚 Natural Language Processing for Business Intelligence")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = HotelReviewSentimentAnalyzer()
    
    # Try to load data from CSV, otherwise use sample data
    print("\n1. Loading hotel review data...")
    try:
        analyzer.load_data('hotel_reviews.csv')
    except:
        analyzer.create_sample_dataset()
    
    # Process reviews
    print("\n2. Processing reviews for sentiment analysis...")
    analyzer.process_all_reviews()
    
    # Generate insights and summary
    print("\n3. Generating comprehensive insights...")
    analyzer.print_summary()
    
    # Create visualizations
    print("\n4. Creating data visualizations...")
    analyzer.create_visualizations()
    
    # Save results
    print("\n5. Saving analysis results...")
    analyzer.save_results()
    
    print("\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print("\n📁 Generated Files:")
    print("   • sentiment_analysis_results.png (visualizations)")
    print("   • hotel_sentiment_analysis_results.csv (detailed results)")
    
    print("\n🎯 Key Features Demonstrated:")
    print("   ✅ TextBlob sentiment analysis implementation")
    print("   ✅ Multi-category service topic extraction")
    print("   ✅ Comprehensive data visualization")
    print("   ✅ Business intelligence insights generation")
    print("   ✅ CSV data export for further analysis")

if __name__ == "__main__":
    main()
