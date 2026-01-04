# Fake News Detection System

This project is a Machine Learning based Fake News Detection system that classifies news articles
as **Fake** or **Real** using Natural Language Processing (NLP) techniques.
The system includes a Flask-based web interface for user interaction.

---

## Project Overview
Fake news has become a major issue in online media platforms.
This project aims to detect fake news articles by analyzing textual content
using machine learning algorithms and NLP techniques.

---

## Features
- Text preprocessing using NLP
- TF-IDF vectorization
- Logistic Regression classifier
- Flask-based web interface
- Real-time news prediction

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Flask
- HTML & CSS

---

## Dataset
This project uses the **Fake and Real News Dataset**.

- The dataset is included in this repository for learning and demonstration purposes.
- Large file size warnings may appear, but the dataset works correctly.

Dataset Source (Kaggle):  
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

---

## Project Structure
```text
fake-news-detection-ml/
│
├── app.py
├── fake_news_model.py
├── requirements.txt
├── data/
│   ├── Fake.csv
│   └── True.csv
├── templates/
│   └── index.html
└── README.md
```


---

## How to Run the Project
1. Clone the repository

git clone https://github.com/Raihan-Raj88/fake-news-detection-ml.git

## Install dependencies
pip install -r requirements.txt

## Run the Flask application
python app.

## Open your browser and visit
http://127.0.0.1:5000

## Output
The system predicts whether the given news is FAKE or REAL
Results are displayed on the web interface

## Future Improvements
Use advanced models like LSTM or BERT
Add prediction confidence score
Deploy the application online

## Author
Md. Raihan Sobhan Raj
Department of Computer Science & Engineering
Green University of Bangladesh

