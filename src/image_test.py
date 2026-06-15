from image_analyzer import (
    ImageAnalyzer
)

analyzer = ImageAnalyzer()

result = analyzer.analyze_image(
    r"C:\Users\Kunal\Downloads\test.jpg"
)

print("\n" + "=" * 60)

print("EXTRACTED TEXT:")
print(result["text"])

print("\nPREDICTION:")
print(result["prediction"])

print(
    f"\nCONFIDENCE: "
    f"{result['confidence']}%"
)

print("\n" + "=" * 60)