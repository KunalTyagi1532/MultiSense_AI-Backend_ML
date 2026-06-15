from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="models/social_media_transformer",
    tokenizer="models/social_media_transformer"
)

print(
    classifier(
        "I love this product"
    )
)