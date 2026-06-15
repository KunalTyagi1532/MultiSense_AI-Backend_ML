import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class TextSentimentAnalyzer:

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_text(
        self,
        text_string: str
    ):

        sentences = re.split(
            r'[.!?]+',
            text_string
        )

        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        results = []

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for sentence in sentences:

            scores = self.analyzer.polarity_scores(
                sentence
            )

            compound_score = scores[
                'compound'
            ]

            if compound_score >= 0.05:

                sentiment = "Positive"
                positive_count += 1

            elif compound_score <= -0.05:

                sentiment = "Negative"
                negative_count += 1

            else:

                sentiment = "Neutral"
                neutral_count += 1

            results.append({

                "sentence":
                sentence,

                "sentiment":
                sentiment,

                "confidence":
                round(
                    abs(
                        compound_score
                    ) * 100,
                    2
                )
            })

        if (
            positive_count > 0
            and negative_count > 0
        ):

            overall_sentiment = "Mixed"

        elif positive_count > negative_count:

            overall_sentiment = "Positive"

        elif negative_count > positive_count:

            overall_sentiment = "Negative"

        else:

            overall_sentiment = "Neutral"

        return {

            "overall_sentiment":
            overall_sentiment,

            "results":
            results
        }

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str
    ):

        return df[
            text_column
        ].apply(
            self.analyze_text
        )