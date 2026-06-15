import pandas as pd


class SlangProcessor:

    slang_dict = {}

    special_phrases = {

        # Positive
        "w company": "Positive",
        "common w": "Positive",
        "huge w": "Positive",
        "massive w": "Positive",
        "my goat": "Positive",
        "the goat": "Positive",
        "goated": "Positive",
        "slaps": "Positive",
        "fire": "Positive",
        "lowkey fire": "Positive",
        "peak": "Positive",
        "we are so back": "Positive",

        # Negative
        "l take": "Negative",
        "common l": "Negative",
        "massive l": "Negative",
        "cooked": "Negative",
        "bro is cooked": "Negative",
        "i am cooked": "Negative",
        "we are cooked": "Negative",
        "mid": "Negative",
        "trash": "Negative",
        "garbage": "Negative",
        "it's over": "Negative",
        "its over": "Negative",
        "joever": "Negative",

        # Sarcasm
        "great, my laptop crashed": "Sarcastic",
        "crashed again": "Sarcastic",
        "fantastic, another bug": "Sarcastic",
        "another bug": "Sarcastic",
        "love waking up to": "Sarcastic",
        "just what i needed": "Sarcastic",
        "another monday": "Sarcastic",
        "thanks for nothing": "Sarcastic",
        "wow great": "Sarcastic",
        "awesome, another": "Sarcastic",
        "love that for me": "Sarcastic"
    }

    @classmethod
    def load(cls):

        df = pd.read_csv(
            "data/slang_dataset.csv"
        )

        cls.slang_dict = dict(
            zip(
                df["slang"].str.lower(),
                df["meaning"]
            )
        )

    @classmethod
    def process(cls, text):

        if not cls.slang_dict:
            cls.load()

        words = text.split()

        processed = []

        for word in words:

            lookup = word.lower()

            if lookup in cls.slang_dict:

                processed.append(
                    cls.slang_dict[lookup]
                )

            else:

                processed.append(
                    word
                )

        return " ".join(
            processed
        )

    @classmethod
    def detect_phrase_sentiment(
        cls,
        text
    ):

        text = text.lower()

        for phrase, sentiment in (
            cls.special_phrases.items()
        ):

            if phrase in text:

                return sentiment

        # Happy crying detection

        if (
            "😭" in text
            and any(
                word in text
                for word in [
                    "selected",
                    "internship",
                    "placed",
                    "offer",
                    "job",
                    "won",
                    "accepted",
                    "passed"
                ]
            )
        ):

            return "Positive"

        return None