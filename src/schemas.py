from pydantic import BaseModel


class SentenceResult(BaseModel):
    sentence: str
    sentiment: str
    confidence: float


class TextResponse(BaseModel):
    text: str
    overall_sentiment: str
    results: list[SentenceResult]


class VisualLabel(BaseModel):
    label: str
    confidence: float


class ImageResponse(BaseModel):
    extracted_text: str
    text_sentiment: str
    text_confidence: float

    content_type: str
    content_confidence: float

    visual_label: str
    visual_confidence: float

    visual_labels: list[VisualLabel]