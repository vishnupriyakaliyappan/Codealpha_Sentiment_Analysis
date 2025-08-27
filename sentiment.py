import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from collections import Counter
import re

# Step 2: Download VADER lexicon and stopwords (only once)
nltk.download('vader_lexicon')
nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 3: Sample Amazon reviews (replace with your CSV if needed)
reviews = [
    "This phone is amazing! Battery lasts all day.",
    "Terrible product. Stopped working after one week.",
    "It’s okay, not the best but not the worst either.",
    "Excellent camera quality, very happy with the purchase!",
    "Waste of money. I want a refund."
]

# Create DataFrame
df = pd.DataFrame(reviews, columns=['Review'])

# Step 4: Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Step 5: Function to determine sentiment
def get_sentiment(text):
    score = sid.polarity_scores(text)['compound']
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Step 6: Apply sentiment analysis
df['Sentiment'] = df['Review'].apply(get_sentiment)

# Step 7: Show results
print("Sentiment Analysis of Each Review:\n")
print(df)

# Step 8: Overall sentiment counts
sentiment_counts = df['Sentiment'].value_counts()
print("\nOverall Sentiment Counts:\n", sentiment_counts)

# Step 9: Extract top positive and negative words
stop_words = set(stopwords.words('english'))
positive_words = []
negative_words = []

for review in df['Review']:
    words = re.findall(r'\b\w+\b', review.lower())
    for word in words:
        if word not in stop_words:
            score = sid.polarity_scores(word)['compound']
            if score >= 0.5:
                positive_words.append(word)
            elif score <= -0.5:
                negative_words.append(word)

# Step 10: Show top words
print("\nTop Positive Words:", Counter(positive_words).most_common(10))
print("Top Negative Words:", Counter(negative_words).most_common(10))


