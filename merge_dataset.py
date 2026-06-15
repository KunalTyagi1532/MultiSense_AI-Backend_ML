import pandas as pd

print("Loading Twitter dataset...")

twitter = pd.read_csv(
    "data/Twitter_Data.csv"
)

twitter = twitter[
    ["clean_text", "category"]
]

twitter = twitter.dropna()

twitter["category"] = (
    twitter["category"]
    .astype(int)
)

twitter["category"] = (
    twitter["category"]
    .replace({
        -1: "Negative",
         0: "Neutral",
         1: "Positive"
    })
)

twitter.columns = [
    "text",
    "label"
]

print(
    f"Twitter rows: {len(twitter)}"
)

print("\nLoading Social dataset...")

social = pd.read_csv(
    "data/social_media_sentiment_train.csv"
)

print(
    f"Social rows: {len(social)}"
)

combined = pd.concat(
    [twitter, social],
    ignore_index=True
)

combined = combined.dropna()

sarcastic = combined[
    combined["label"] == "Sarcastic"
]

sarcastic_boosted = pd.concat(
    [sarcastic] * 10,
    ignore_index=True
)

combined = pd.concat(
    [
        combined[
            combined["label"] != "Sarcastic"
        ],
        sarcastic_boosted
    ],
    ignore_index=True
)

print(
    f"\nCombined rows: {len(combined)}"
)

print(
    "\nClass Distribution:"
)

print(
    combined["label"]
    .value_counts()
)

combined.to_csv(
    "data/combined_dataset.csv",
    index=False
)

print(
    "\nSaved to data/combined_dataset.csv"
)