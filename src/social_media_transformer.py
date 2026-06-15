import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

import evaluate


class SocialMediaTransformerTrainer:

    def train(self):

        print("\n" + "=" * 50)
        print("LOADING DATASET")
        print("=" * 50)

        df = pd.read_csv(
            "data/combined_dataset.csv"
        )

        print("\nDataset Shape:")
        print(df.shape)

        print("\nFirst 5 Rows:")
        print(df.head())

        print("\nOriginal Label Distribution:")
        print(df["label"].value_counts())

        label_map = {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2,
            "Sarcastic": 3
        }

        id2label = {
            0: "Negative",
            1: "Neutral",
            2: "Positive",
            3: "Sarcastic"
        }

        label2id = {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2,
            "Sarcastic": 3
        }

        df["label"] = df["label"].map(
            label_map
        )

        print("\nMissing Labels:")
        print(df["label"].isnull().sum())

        print("\nEncoded Label Distribution:")
        print(df["label"].value_counts())

        dataset = Dataset.from_pandas(
            df
        )

        MODEL_NAME = (
            "distilbert-base-uncased"
        )

        print("\nLoading Tokenizer...")

        tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_NAME
            )
        )

        def tokenize(batch):

            return tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=64
            )

        print("\nTokenizing Dataset...")

        dataset = dataset.map(
            tokenize,
            batched=True
        )

        dataset = (
            dataset.train_test_split(
                test_size=0.2,
                seed=42
            )
        )

        print("\nLoading DistilBERT...")

        model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                MODEL_NAME,
                num_labels=4,
                id2label=id2label,
                label2id=label2id
            )
        )

        accuracy_metric = (
            evaluate.load(
                "accuracy"
            )
        )

        def compute_metrics(eval_pred):

            logits, labels = eval_pred

            predictions = np.argmax(
                logits,
                axis=-1
            )

            return accuracy_metric.compute(
                predictions=predictions,
                references=labels
            )

        training_args = TrainingArguments(
            output_dir="social_model",

            num_train_epochs=2,

            per_device_train_batch_size=8,

            per_device_eval_batch_size=8,

            eval_strategy="epoch",

            save_strategy="epoch",

            logging_steps=100,

            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,

            train_dataset=dataset["train"],

            eval_dataset=dataset["test"],

            compute_metrics=
            compute_metrics
        )

        print("\nStarting Training...\n")

        trainer.train()

        print("\nEvaluating Model...\n")

        results = trainer.evaluate()

        print("\nFinal Results:")
        print(results)

        model.save_pretrained(
            "models/social_media_transformer"
        )

        tokenizer.save_pretrained(
            "models/social_media_transformer"
        )

        print(
            "\nTransformer saved successfully."
        )

        print(
            "\nModel Label Mapping:"
        )

        print(
            model.config.id2label
        )