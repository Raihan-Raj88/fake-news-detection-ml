import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords (only first time)
nltk.download('stopwords')

print("Loading dataset...")

# Load datasets
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Add labels
fake_df["label"] = 0
true_df["label"] = 1

# Combine and shuffle
df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine title + text
df["content"] = df["title"] + " " + df["text"]

# Load stopwords once (IMPORTANT FIX)
stop_words = set(stopwords.words('english'))

# Text cleaning function (FAST)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

print("Cleaning text...")
df["content"] = df["content"].apply(clean_text)

# Features and labels
X = df["content"]
y = df["label"]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Vectorizing text...")
vectorizer = TfidfVectorizer(max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

# Prediction function
def predict_news(title, text):
    combined = title + " " + text
    combined = clean_text(combined)
    vec = vectorizer.transform([combined])
    pred = model.predict(vec)[0]
    return "REAL NEWS" if pred == 1 else "FAKE NEWS"

# ---------------- USER INPUT LOOP ----------------
print("\n===== FAKE NEWS DETECTOR =====")
print("Type 'exit' to quit\n")

while True:
    title = input("Enter News Title: ")
    if title.lower() == "exit":
        break

    text = input("Enter News Description: ")
    if text.lower() == "exit":
        break

    result = predict_news(title, text)
    print("\nPrediction:", result)
    print("-" * 40)

print("Program ended.")
