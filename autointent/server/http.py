"""FastAPI application for AutoIntent pipeline inference."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from autointent import Pipeline
from autointent.custom_types import ListOfLabelsWithOOS


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AUTOINTENT_")
    path: str = Field(..., description="Path to the optimized pipeline assets")
    host: str = "127.0.0.1"
    port: int = 8013


class PredictRequest(BaseModel):
    """Request model for the predict endpoint."""

    utterances: list[str] = Field(..., description="List of text utterances to classify")


class PredictResponse(BaseModel):
    """Response model for the predict endpoint."""

    predictions: ListOfLabelsWithOOS = Field(..., description="List of predicted class labels")


settings = Settings()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_pipeline() -> Pipeline:
    """Load the optimized pipeline from disk."""
    pipeline_path = Path(settings.path)
    if not pipeline_path.exists():
        msg = f"Pipeline path does not exist: {pipeline_path}"
        logger.error(msg)
        raise HTTPException(status_code=404, detail=msg)

    try:
        msg = f"Loading pipeline from: {pipeline_path}"
        logger.info(msg)
        pipeline = Pipeline.load(pipeline_path)
        logger.info("Pipeline loaded successfully")

    except Exception:
        logger.exception("Failed to load pipeline")
        raise
    else:
        return pipeline


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Load pipe."""
    load_pipeline()
    yield


app = FastAPI(
    title="AutoIntent Pipeline API",
    description="API for serving AutoIntent predictions",
    version="0.0.1",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict")
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict class labels for the given utterances.

    Args:
        request: Request containing list of utterances to classify

    Returns:
        Response containing predicted class labels
    """
    current_pipeline = load_pipeline()

    if not request.utterances:
        return PredictResponse(predictions=[])

    predictions = current_pipeline.predict(request.utterances)

    return PredictResponse(predictions=predictions)


def main() -> None:
    """Main entry point for the HTTP server."""
    import uvicorn

    uvicorn.run(
        "autointent.server.http:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
