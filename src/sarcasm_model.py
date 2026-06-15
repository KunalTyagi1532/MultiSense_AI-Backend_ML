import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SarcasmModel:

    def train(self):

        df1 = pd.read_json(
            "data/Sarcasm_Headlines_Dataset.json",
            lines=True
        )

        df2 = pd.read_json(
            "data/Sarcasm_Headlines_Dataset_v2.json",
            lines=True
        )

        df = pd.concat(
            [df1, df2],
            ignore_index=True
        )

        print(f"Total Samples: {len(df)}")

        X = df["headline"]
        y = df["is_sarcastic"]

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
            random_state=42,
            stratify=y
        )

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            f"Sarcasm Accuracy: {accuracy:.4f}"
        )

        joblib.dump(
            (model, vectorizer),
            "models/sarcasm_model.pkl"
        )

        print(
            "Sarcasm model saved successfully."
        )

    def predict(self, text):

        model, vectorizer = joblib.load(
            "models/sarcasm_model.pkl"
        )

        text_vector = vectorizer.transform(
            [text]
        )

        prediction = model.predict(
            text_vector
        )[0]

        probabilities = model.predict_proba(
            text_vector
        )

        confidence = (
            max(probabilities[0]) * 100
        )

        return prediction, confidence