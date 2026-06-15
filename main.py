from src.social_media_transformer_predictor import (
    SocialMediaTransformerPredictor
)

predictor = (
    SocialMediaTransformerPredictor()
)

tests = [
    "W company",
    "L take",
    "bro is cooked 💀",
    "nah bro that's wild",
    "common W",
    "my goat",
    "we are so back",
    "it's over",
    "Great, my laptop crashed again",
    "Fantastic, another bug",
    "Love waking up to 20 assignments",
    "Just what I needed, a flat tire",
    "I got selected 😭😭😭",
    "This movie slaps 🔥",
    "This update is fire 🔥"
]

for text in tests:

    label, confidence = (
        predictor.predict(text)
    )

    print("\n" + "=" * 60)

    print(
        f"TEXT: {text}"
    )

    print(
        f"PREDICTION: {label}"
    )

    print(
        f"CONFIDENCE: {confidence}%"
    )