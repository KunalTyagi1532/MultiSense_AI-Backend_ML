import easyocr

from src.social_media_transformer_predictor import (
    SocialMediaTransformerPredictor
)


class ImageAnalyzer:

    def __init__(self):

        self.reader = easyocr.Reader(
            ['en']
        )

        self.predictor = (
            SocialMediaTransformerPredictor()
        )

    def analyze_image(
        self,
        image_path
    ):

        try:

            extracted_text = (
                self.reader.readtext(
                    image_path,
                    detail=0
                )
            )

        except Exception as e:

            return {
                "text": "",
                "prediction": "OCR Error",
                "confidence": 0,
                "error": str(e)
            }

        text = " ".join(
            extracted_text
        )

        if not text.strip():

            return {
                "text": "",
                "prediction":
                "No text found",
                "confidence":
                0
            }

        try:

            sentiment, confidence = (
                self.predictor.predict(
                    text
                )
            )

            return {
                "text": text,
                "prediction":
                sentiment,
                "confidence":
                confidence
            }

        except Exception as e:

            return {
                "text": text,
                "prediction":
                "Prediction Error",
                "confidence":
                0,
                "error":
                str(e)
            }