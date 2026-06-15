import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SentimentMLModel:

    def train(self):

        df = pd.read_csv("data/Twitter_Data.csv")

        df = df[["clean_text", "category"]]

        df = df.dropna()

        label_map = {
            -1: "negative",
             0: "neutral",
             1: "positive"
        }

        df["category"] = df["category"].map(label_map)
        print("\nClass Distribution:")
        print(df["category"].value_counts())

        X = df["clean_text"]
        y = df["category"]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=15000,
            ngram_range=(1, 2)
        )

        X_vectorized = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized,
            y,
            test_size=0.2,
            random_state=42
        )

        model = LogisticRegression(
            max_iter=1000
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(f"Accuracy: {accuracy:.4f}")

        joblib.dump(
            (model, vectorizer),
            "models/sentiment_model.pkl"
        )

        print("Model saved successfully.")

    def predict(self, text):

        model, vectorizer = joblib.load(
            "models/sentiment_model.pkl"
        )

        text_vector = vectorizer.transform([text])

        prediction = model.predict(text_vector)

        probabilities = model.predict_proba(text_vector)

        confidence = max(probabilities[0]) * 100

        return prediction[0], confidence