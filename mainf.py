# Phishing Email Classifier using CEAS_08.csv - Random Forest Only
# Google Codelabs-style Tutorial
# Dataset: CEAS_08.csv with columns: Sender Email, Receiver Email, Date Time, Subject, Body, Label, URL

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import warnings
warnings.filterwarnings('ignore')
import pickle

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

# Download NLTK resources
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    stop_words = set(stopwords.words('english'))
except:
    print("NLTK resources download failed, using empty stopwords set")
    stop_words = set()

# Step 2: Load and Explore Data with Error Handling
def load_csv_robust(filename):
    """Robust CSV loading to handle malformed data"""
    try:
        df = pd.read_csv(filename,
                         encoding='latin1',  # Handle special characters
                         quoting=csv.QUOTE_ALL,  # Quote all fields
                         on_bad_lines='skip',  # Skip malformed rows
                         low_memory=False)  # Disable low-memory mode for complex files
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        print("Attempting to read with alternative method...")

        # Fallback: Read file line-by-line and parse manually
        rows = []
        with open(filename, encoding='latin1') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_ALL)
            header = next(reader)  # Get header
            for i, row in enumerate(reader):
                try:
                    if len(row) == len(header):  # Ensure correct number of columns
                        rows.append(row)
                    else:
                        print(f"Skipping row {i+2}: incorrect column count")
                except Exception as e:
                    print(f"Skipping row {i+2}: parsing error - {str(e)}")
        return pd.DataFrame(rows, columns=header)

# Load the dataset
df = load_csv_robust('CEAS_08.csv')

# Display basic info
print("Dataset Info:")
print(df.info())
print("\nSample Data:")
print(df.head())
print("\nClass Distribution:")
print(df['label'].value_counts())

# Step 3: Preprocess Data
# Handle missing values - using the correct column names from your data
df['subject'] = df['subject'].fillna('')
df['body'] = df['body'].fillna('')
df['urls'] = df['urls'].fillna('')
df['sender'] = df.get('sender', df.get('from', '')).fillna('unknown')
df['receiver'] = df.get('receiver', df.get('to', '')).fillna('unknown')
df['date'] = df['date'].fillna('unknown')

# Clean text data
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    if stop_words:  # Only tokenize if NLTK is available
        try:
            tokens = word_tokenize(text)
            tokens = [t for t in tokens if t not in stop_words]
            return ' '.join(tokens)
        except:
            return text
    return text

df['subject'] = df['subject'].apply(clean_text)
df['body'] = df['body'].apply(clean_text)

# Feature: Extract URL characteristics
def extract_url_features(url):
    if not isinstance(url, str):
        url = str(url)
    if url == '' or url == 'unknown' or url == 'nan':
        return 0, 0, 0
    has_ip = 1 if re.search(r'https?://(?:\d{1,3}\.){3}\d{1,3}', url) else 0
    has_punycode = 1 if re.search(r'https?://xn--', url) else 0
    risky_tld = 1 if re.search(r'https?://[^\s]+\.(zip|mov|info|ru|cn)(/|$)', url) else 0
    return has_ip, has_punycode, risky_tld

# Apply extract_url_features to 'urls' column
url_features = df['urls'].apply(extract_url_features)
url_features_df = pd.DataFrame(url_features.tolist(), columns=['has_ip', 'has_punycode', 'risky_tld'])
df = pd.concat([df, url_features_df], axis=1)

# Feature: Sender domain analysis
df['sender_domain'] = df['sender'].apply(lambda x: x.split('@')[-1] if '@' in str(x) else 'unknown')

# Feature: Date Time (hour of day)
def extract_hour_safe(date_str):
    """Safely extract hour from date string"""
    try:
        if pd.isna(date_str) or date_str == 'unknown':
            return 12  # Default to noon
        
        # Try to parse the date
        parsed_date = pd.to_datetime(date_str, errors='coerce')
        if pd.isna(parsed_date):
            return 12  # Default to noon if parsing fails
        
        return parsed_date.hour
    except:
        return 12  # Default to noon if any error occurs

df['hour'] = df['date'].apply(extract_hour_safe)

# Step 4: Feature Engineering
# Combine Subject and Body for TF-IDF
df['text'] = df['subject'] + ' ' + df['body']

# Ensure label is numeric and convert to integer
df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

# TF-IDF Vectorization
print("\nCreating TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', max_df=0.8, min_df=2)
X_tfidf = tfidf.fit_transform(df['text'])

# Combine TF-IDF with metadata features
X_meta = df[['has_ip', 'has_punycode', 'risky_tld', 'hour']].values
X = np.hstack((X_tfidf.toarray(), X_meta))

# Labels
y = df['label'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 5: Train Random Forest Model
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate Random Forest
print("\nRandom Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")

# Handle ROC-AUC calculation safely
try:
    y_pred_proba = rf.predict_proba(X_test)
    if y_pred_proba.shape[1] > 1:
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")
    else:
        print("ROC-AUC: Cannot calculate (only one class predicted)")
except:
    print("ROC-AUC: Cannot calculate")

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Feature importance
feature_names = list(tfidf.get_feature_names_out()) + ['has_ip', 'has_punycode', 'risky_tld', 'hour']
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Step 6: Save Models
print("\nSaving models...")

with open('rf_phishing_classifier.pkl', 'wb') as f:
    pickle.dump(rf, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Models saved successfully!")

# Step 7: Create a simple prediction function
def predict_phishing(email_text, subject_text=""):
    """
    Predict if an email is phishing or not
    """
    # Clean the text
    combined_text = clean_text(subject_text + " " + email_text)
    
    # Vectorize
    text_features = tfidf.transform([combined_text])
    
    # Add dummy metadata features (you would extract these from the actual email)
    meta_features = np.array([[0, 0, 0, 12]])  # has_ip, has_punycode, risky_tld, hour
    
    # Combine features
    features = np.hstack((text_features.toarray(), meta_features))
    
    # Predict
    prediction = rf.predict(features)[0]
    probability = rf.predict_proba(features)[0]
    
    return {
        'is_phishing': bool(prediction),
        'confidence': max(probability),
        'probabilities': {'legitimate': probability[0], 'phishing': probability[1]}
    }

# Test the prediction function
test_email = "Congratulations! You've won $1,000,000! Click here to claim your prize now!"
result = predict_phishing(test_email)
print(f"\nTest prediction: {result}")

print("\nPhishing Email Classifier training complete!")
