from src.multimodal_analyzer import (
    MultimodalAnalyzer
)

analyzer = (
    MultimodalAnalyzer()
)

result = analyzer.analyze_image(
    r"C:\Users\Kunal\Pictures\test\test2.jpg"
)

print("\n" + "=" * 60)

for key, value in result.items():

    print(
        f"{key}: {value}"
    )

print("\n" + "=" * 60)