from transformers import pipeline

class EmotionAnalyzer:

    def __init__(self):
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )

    def predict(self, text):

        result = self.classifier(text)[0]

        best = max(result, key=lambda x: x["score"])

        return (
            best["label"],
            best["score"] * 100
        )