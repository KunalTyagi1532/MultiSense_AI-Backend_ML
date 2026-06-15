import re

from transformers import pipeline

from src.slang_processor import (
    SlangProcessor
)


class SocialMediaTransformerPredictor:

    def __init__(self):

        self.classifier = pipeline(
            "text-classification",
            model="models/social_media_transformer",
            tokenizer="models/social_media_transformer"
        )

    def _predict_single(
        self,
        text
    ):

        phrase_sentiment = (
            SlangProcessor
            .detect_phrase_sentiment(
                text
            )
        )

        if phrase_sentiment:

            return {
                "sentence": text,
                "sentiment": phrase_sentiment,
                "confidence": 99.0
            }

        processed_text = (
            SlangProcessor.process(
                text
            )
        )

        result = self.classifier(
            processed_text
        )[0]

        return {
            "sentence": text,
            "sentiment": result["label"],
            "confidence": round(
                result["score"] * 100,
                2
            )
        }

    def predict(
        self,
        text
    ):

        sentences = re.split(
            r'[.!?]+',
            text
        )

        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        if len(sentences) <= 1:

            result = self._predict_single(
                text
            )

            return {
                "overall_sentiment":
                result["sentiment"],

                "results":
                [result]
            }

        results = []

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for sentence in sentences:

            prediction = (
                self._predict_single(
                    sentence
                )
            )

            sentiment = (
                prediction[
                    "sentiment"
                ]
                .lower()
            )

            if sentiment == "positive":
                positive_count += 1

            elif sentiment == "negative":
                negative_count += 1

            else:
                neutral_count += 1

            results.append(
                prediction
            )

        if (
            positive_count > 0
            and negative_count > 0
        ):

            overall_sentiment = "Mixed"

        elif (
            positive_count >
            negative_count
        ):

            overall_sentiment = "Positive"

        elif (
            negative_count >
            positive_count
        ):

            overall_sentiment = "Negative"

        else:

            overall_sentiment = "Neutral"

        return {
            "overall_sentiment":
            overall_sentiment,

            "results":
            results
        }