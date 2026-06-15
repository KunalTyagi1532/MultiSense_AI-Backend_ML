from src.image_analyzer import (
    ImageAnalyzer
)

from src.visual_sentiment import (
    VisualSentimentAnalyzer
)


class MultimodalAnalyzer:

    def __init__(self):

        self.ocr_analyzer = (
            ImageAnalyzer()
        )

        self.visual_analyzer = (
            VisualSentimentAnalyzer()
        )

    def analyze_image(
        self,
        image_path
    ):

        result = {}

        # OCR + Text Sentiment

        ocr_result = (
            self.ocr_analyzer
            .analyze_image(
                image_path
            )
        )

        result[
            "extracted_text"
        ] = ocr_result.get(
            "text",
            ""
        )

        result[
            "text_sentiment"
        ] = ocr_result.get(
            "prediction",
            ""
        )

        result[
            "text_confidence"
        ] = ocr_result.get(
            "confidence",
            0
        )

        # Visual Analysis

        visual_result = (
            self.visual_analyzer
            .analyze(
                image_path
            )
        )

        result[
            "content_type"
        ] = visual_result.get(
            "content_type",
            ""
        )

        result[
            "content_confidence"
        ] = visual_result.get(
            "content_confidence",
            0
        )

        result[
            "visual_labels"
        ] = visual_result.get(
            "visual_labels",
            []
        )

        if result[
            "visual_labels"
        ]:

            top_label = (
                result[
                    "visual_labels"
                ][0]
            )

            result[
                "visual_label"
            ] = top_label[
                "label"
            ]

            result[
                "visual_confidence"
            ] = top_label[
                "confidence"
            ]

        else:

            result[
                "visual_label"
            ] = "Unknown"

            result[
                "visual_confidence"
            ] = 0

        return result