from flask import Flask, render_template, request
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

app = Flask(__name__)

# ---------- LOAD DATA ----------
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df["content"] = df["title"] + " " + df["text"]

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["content"] = df["content"].apply(clean_text)

X = df["content"]
y = df["label"]

# ---------- TRAIN MODEL ----------
vectorizer = TfidfVectorizer(max_df=0.7)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# ---------- WEB UI ----------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    title = ""
    text = ""

    if request.method == "POST":
        title = request.form.get("title")
        text = request.form.get("text")

        combined = clean_text(title + " " + text)
        vec = vectorizer.transform([combined])
        pred = model.predict(vec)[0]

        result = "REAL NEWS ✅" if pred == 1 else "FAKE NEWS ❌"

    return render_template(
        "index.html",
        result=result,
        title=title,
        text=text
    )

if __name__ == "__main__":
    app.run(debug=True)
