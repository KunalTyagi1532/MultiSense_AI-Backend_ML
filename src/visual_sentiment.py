from transformers import pipeline
from PIL import Image


class VisualSentimentAnalyzer:

    def __init__(self):

        self.visual_classifier = pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32"
        )

        self.content_classifier = pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32"
        )

        self.visual_labels = [

            # People
            "person",
            "man",
            "woman",
            "group of people",

            # Fitness
            "bodybuilder",
            "fitness model",
            "gym workout",
            "muscular physique",
            "athletic person",

            # Emotions
            "happy person",
            "sad person",
            "confident person",
            "angry person",

            # Social Media
            "selfie",
            "social media post",
            "influencer content",
            "motivational content",

            # Lifestyle
            "indoor scene",
            "outdoor scene",
            "office environment",
            "home environment",

            # Food
            "food image",
            "restaurant meal",

            # Nature
            "landscape",
            "animal",

            # Text Content
            "motivational quote",
            "meme",
            "advertisement"
        ]

        self.content_labels = [
            "fitness content",
            "travel content",
            "food content",
            "meme content",
            "motivational content",
            "educational content",
            "fashion content",
            "lifestyle content"
        ]

    def analyze(
        self,
        image_path
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        visual_results = (
            self.visual_classifier(
                image,
                candidate_labels=
                self.visual_labels
            )
        )

        content_results = (
            self.content_classifier(
                image,
                candidate_labels=
                self.content_labels
            )
        )

        top_visual = [
            {
                "label":
                item["label"],

                "confidence":
                round(
                    item["score"] * 100,
                    2
                )
            }
            for item in visual_results[:3]
        ]

        top_content = (
            content_results[0]
        )

        return {

            "content_type":
            top_content["label"],

            "content_confidence":
            round(
                top_content["score"] * 100,
                2
            ),

            "visual_labels":
            top_visual
        }