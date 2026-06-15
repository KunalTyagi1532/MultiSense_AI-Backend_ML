from src.transformer_sentiment import TransformerSentimentAnalyzer
from src.emotion_analyzer import EmotionAnalyzer
from src.emoji_processor import EmojiProcessor
from src.sarcasm_model import SarcasmModel


class SocialMediaAnalyzer:

    def __init__(self):

        self.sentiment_model = TransformerSentimentAnalyzer()
        self.emotion_model = EmotionAnalyzer()
        self.sarcasm_model = SarcasmModel()

    def analyze(self, text):

        processed_text = EmojiProcessor.process(
            text
        )

        sentiment, sentiment_confidence = (
            self.sentiment_model.predict(
                processed_text
            )
        )

        emotion, emotion_confidence = (
            self.emotion_model.predict(
                processed_text
            )
        )

        sarcasm, sarcasm_confidence = (
            self.sarcasm_model.predict(
                processed_text
            )
        )

        if emotion in ["sadness", "anger", "fear"]:
            final_sentiment = "negative"

        elif emotion in ["joy", "love"]:
            final_sentiment = "positive"

        else:
            final_sentiment = sentiment

        return {
            "original_text": text,
            "processed_text": processed_text,
            "sentiment": final_sentiment,
            "sentiment_confidence": round(
                sentiment_confidence, 2
            ),
            "emotion": emotion,
            "emotion_confidence": round(
                emotion_confidence, 2
            ),
            "sarcasm": bool(sarcasm),
            "sarcasm_confidence": round(
                sarcasm_confidence, 2
            )
        }