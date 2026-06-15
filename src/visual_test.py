from visual_sentiment import (
    VisualSentimentAnalyzer
)

analyzer = (
    VisualSentimentAnalyzer()
)

result = analyzer.analyze(
    r"C:\Users\Kunal\Pictures\test\test2.jpg"
)

for item in result[:5]:

    print(
        item["label"],
        round(
            item["score"] * 100,
            2
        )
    )