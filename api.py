from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

import shutil
import os
import uuid

from src.schemas import (
    TextResponse,
    ImageResponse
)

from src.social_media_transformer_predictor import (
    SocialMediaTransformerPredictor
)

from src.multimodal_analyzer import (
    MultimodalAnalyzer
)

app = FastAPI(
    title="Social Media Analyzer API",
    description="AI-powered multimodal social media analyzer",
    version="1.0.0"
)

text_analyzer = (
    SocialMediaTransformerPredictor()
)

image_analyzer = (
    MultimodalAnalyzer()
)

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

MAX_IMAGE_SIZE_MB = 20

MAX_IMAGE_SIZE = (
    MAX_IMAGE_SIZE_MB
    * 1024
    * 1024
)


@app.get("/")
def home():

    return {
        "message":
        "Social Media Analyzer API"
    }


@app.get("/health")
def health():

    return {
        "status":
        "healthy"
    }


@app.post(
    "/analyze-text",
    response_model=TextResponse
)
def analyze_text(
    text: str
):

    if not text.strip():

        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    analysis = (
        text_analyzer.predict(
            text
        )
    )

    return TextResponse(
        text=text,

        overall_sentiment=
        analysis[
            "overall_sentiment"
        ],

        results=
        analysis[
            "results"
        ]
    )


@app.post(
    "/analyze-image",
    response_model=ImageResponse
)
async def analyze_image(
    file: UploadFile = File(...)
):

    extension = (
        os.path.splitext(
            file.filename
        )[1].lower()
    )

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail="Unsupported image format"
        )

    file_path = (
        f"temp_"
        f"{uuid.uuid4()}_"
        f"{file.filename}"
    )

    try:

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        if (
            os.path.getsize(
                file_path
            )
            > MAX_IMAGE_SIZE
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Image exceeds "
                    f"{MAX_IMAGE_SIZE_MB} MB limit"
                )
            )

        result = (
            image_analyzer
            .analyze_image(
                file_path
            )
        )

        return ImageResponse(

            extracted_text=result.get(
                "extracted_text",
                ""
            ),

            text_sentiment=result.get(
                "text_sentiment",
                ""
            ),

            text_confidence=result.get(
                "text_confidence",
                0
            ),

            content_type=result.get(
                "content_type",
                ""
            ),

            content_confidence=result.get(
                "content_confidence",
                0
            ),

            visual_label=result.get(
                "visual_label",
                ""
            ),

            visual_confidence=result.get(
                "visual_confidence",
                0
            ),

            visual_labels=result.get(
                "visual_labels",
                []
            )
        )

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    finally:

        if os.path.exists(
            file_path
        ):

            os.remove(
                file_path
            )