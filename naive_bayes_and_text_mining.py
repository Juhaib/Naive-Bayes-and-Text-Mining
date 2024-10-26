import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import nltk

# Download stopwords if necessary
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_csv("blogs.csv")

# Function to extract main content, excluding metadata
def extract_content(text):
    # Split text at first empty line (assumes body follows metadata)
    content = re.split(r'\n\s*\n', text, maxsplit=1)
    return content[1] if len(content) > 1 else content[0]

# Apply the extraction function
df['Cleaned_Data'] = df['Data'].apply(extract_content)

# Data Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Apply text cleaning
df['Cleaned_Data'] = df['Cleaned_Data'].apply(preprocess_text)

# Feature Extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['Cleaned_Data']).toarray()
y = df['Labels']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes Model for Text Classification
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Sentiment Analysis
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Cleaned_Data'].apply(get_sentiment)

# Sentiment Distribution Analysis
sentiment_counts = df.groupby('Labels')['Sentiment'].value_counts(normalize=True).unstack()
print("\nSentiment Distribution Across Categories:\n", sentiment_counts)

# Save Results and Cleaned Dataset (Optional)
df.to_csv("blogs_categories_with_sentiments.csv", index=False)
print("\nResults saved to blogs_categories_with_sentiments.csv")
