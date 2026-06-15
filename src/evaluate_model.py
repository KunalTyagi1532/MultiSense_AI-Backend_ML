import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

MODEL_PATH = "models/social_media_transformer"

# Load model
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(MODEL_PATH)
)

# Load dataset
df = pd.read_csv(
    "data/combined_dataset.csv"
)

label_map = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
    "Sarcastic": 3
}

df["label"] = df["label"].map(
    label_map
)

dataset = Dataset.from_pandas(df)

def tokenize(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=64
    )

dataset = dataset.map(
    tokenize,
    batched=True
)

dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

trainer = Trainer(
    model=model
)

predictions = trainer.predict(
    dataset["test"]
)

preds = np.argmax(
    predictions.predictions,
    axis=1
)

labels = predictions.label_ids

print("\nClassification Report:\n")

print(
    classification_report(
        labels,
        preds,
        target_names=[
            "Negative",
            "Neutral",
            "Positive",
            "Sarcastic"
        ]
    )
)

cm = confusion_matrix(
    labels,
    preds
)

print("\nConfusion Matrix:\n")
print(cm)

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.colorbar()

plt.xticks(
    [0,1,2,3],
    ["Neg","Neu","Pos","Sar"]
)

plt.yticks(
    [0,1,2,3],
    ["Neg","Neu","Pos","Sar"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Confusion Matrix"
)

plt.show()