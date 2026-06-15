from transformers import pipeline


class TransformerSentimentAnalyzer:

    def __init__(self):

        self.model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

    def predict(self, text):

        result = self.model(text)[0]

        sentiment = result["label"]
        confidence = result["score"] * 100

        return sentiment, confidence