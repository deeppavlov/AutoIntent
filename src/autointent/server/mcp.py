"""MCP Server for AutoIntent."""

import logging
import math
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from autointent import Dataset, Pipeline
from autointent.custom_types import LabelWithOOS, ListOfLabelsWithOOS, Split
from autointent.schemas import Intent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AUTOINTENT_")
    path: str = Field(..., description="Path to the optimized pipeline assets")
    transport: Literal["stdio", "http"] = "stdio"
    host: str = "127.0.0.1"
    port: int = 8012


class PredictOutput(BaseModel):
    """Output model for the predict tool."""

    predictions: ListOfLabelsWithOOS = Field(..., description="List of predicted class labels")


class PaginationParams(BaseModel):
    """Params to perform pagination."""

    page: int = Field(default=1, description="Page number (1-indexed)", ge=1)
    page_size: int = Field(default=20, description="Number of classes per page", ge=1, le=100)


class PaginationInfo(BaseModel):
    """Information about data pages."""

    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")


class ClassesOutput(BaseModel):
    """Output model for the classes tool."""

    classes: list[Intent] = Field(..., description="List of class information")
    pagination_info: PaginationInfo


class DataSample(BaseModel):
    """Single training sample."""

    id: int
    text: str
    label: LabelWithOOS


class TrainDataOutput(BaseModel):
    """Output model for the train_data tool."""

    samples: list[DataSample] = Field(..., description="List of training samples")
    pagination_info: PaginationInfo


logger = logging.getLogger(__name__)


def get_settings() -> Settings:
    """Get or create settings instance."""
    return Settings()


@lru_cache(maxsize=1)
def load_pipeline() -> Pipeline:
    """Load the optimized pipeline from disk."""
    settings = get_settings()
    pipeline_path = Path(settings.path)
    if not pipeline_path.exists():
        msg = f"Pipeline path does not exist: {pipeline_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

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


@lru_cache(maxsize=1)
def load_dataset() -> Dataset:
    """Load the training dataset from the pipeline directory."""
    settings = get_settings()
    pipeline_path = Path(settings.path)
    dataset_path = pipeline_path / "dataset.json"

    if not dataset_path.exists():
        msg = f"Dataset file does not exist: {dataset_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        msg = f"Loading dataset from: {dataset_path}"
        logger.info(msg)
        dataset = Dataset.from_json(dataset_path)
        logger.info("Dataset loaded successfully")
    except Exception:
        logger.exception("Failed to load dataset")
        raise
    else:
        return dataset


@asynccontextmanager
async def lifespan(_: FastMCP) -> AsyncGenerator[None, None]:
    """Lifespan manager for the MCP server."""
    logger.info("Starting AutoIntent MCP server...")

    # Preload pipeline and dataset
    try:
        load_pipeline()
        load_dataset()
        logger.info("AutoIntent MCP server ready")
    except Exception:
        logger.exception("Failed to initialize server")
        raise

    yield

    logger.info("Shutting down AutoIntent MCP server")


# Create the FastMCP server
mcp = FastMCP(
    name="AutoIntent MCP Server",
    instructions="""
    AutoIntent MCP Server provides text classification capabilities through a trained AutoIntent pipeline.

    The server loads both the trained pipeline and the original dataset from the pipeline directory.
    Set AUTOINTENT_PATH environment variable to the path containing the trained pipeline assets.

    Available tools:
    - predict: Classify text utterances using the trained model
    - classes: Get paginated list of all available classes/intents with their metadata
    - train_data: Access training samples with pagination and filtering options

    This server allows LLMs to leverage AutoIntent's powerful text classification models for text classification tasks.
    """,
    lifespan=lifespan,
)


@mcp.tool()
def predict(utterances: list[str]) -> PredictOutput:
    """Classify text utterances using the trained AutoIntent pipeline.

    Args:
        utterances: List of text utterances to classify

    Returns:
        Predictions for each utterance
    """
    pipeline = load_pipeline()

    if not utterances:
        return PredictOutput(predictions=[])

    try:
        predictions = pipeline.predict(utterances)
        return PredictOutput(predictions=predictions)
    except Exception:
        logger.exception("Failed to make predictions")
        raise


@mcp.tool()
def classes(page: int = 1, page_size: int = 20) -> ClassesOutput:
    """Get paginated list of all available classes/intents with their metadata.

    Args:
        page: Page number (1-indexed)
        page_size: Number of classes per page

    Returns:
        Paginated list of classes with metadata
    """
    dataset = load_dataset()

    # Create pagination params object
    pagination_params = PaginationParams(page=page, page_size=page_size)

    total_classes = dataset.n_classes
    start_idx, end_idx = _calculate_page_bounds(
        params=pagination_params,
        total_items=total_classes,
    )

    # Get the paginated slice of intents
    paginated_intents = dataset.intents[start_idx:end_idx]

    pagination_info = PaginationInfo(
        total_items=total_classes, total_pages=math.ceil(total_classes / pagination_params.page_size)
    )

    return ClassesOutput(classes=paginated_intents, pagination_info=pagination_info)


@mcp.tool()
def train_data(page: int = 1, page_size: int = 20, class_filter: list[int] | None = None) -> TrainDataOutput:
    """Access training samples with pagination and filtering options.

    Args:
        page: Page number (1-indexed)
        page_size: Number of samples per page
        class_filter: Filter by specific class IDs

    Returns:
        Paginated and filtered training samples
    """
    dataset = load_dataset()
    # Handle both 'train' and 'train_0' splits similar to how dataset.multilabel property works
    train_split = Split.TRAIN if Split.TRAIN in dataset else f"{Split.TRAIN}_0"
    split_data = dataset[train_split]

    # Convert to DataSample list
    samples_list: list[DataSample] = []
    for i in range(len(split_data)):
        sample = DataSample(
            id=i, text=split_data[i][dataset.utterance_feature], label=split_data[i][dataset.label_feature]
        )
        samples_list.append(sample)

    # Apply class filtering if specified
    if class_filter is not None:
        samples_list = _filter_samples_by_class(samples_list, class_filter, dataset.multilabel)

    total_samples = len(samples_list)

    # Create pagination params object
    pagination_params = PaginationParams(page=page, page_size=page_size)
    start_idx, end_idx = _calculate_page_bounds(
        params=pagination_params,
        total_items=total_samples,
    )

    # Get paginated samples
    paginated_samples = samples_list[start_idx:end_idx]

    pagination_info = PaginationInfo(
        total_items=total_samples,
        total_pages=math.ceil(total_samples / pagination_params.page_size) if total_samples > 0 else 1,
    )

    return TrainDataOutput(samples=paginated_samples, pagination_info=pagination_info)


def _calculate_page_bounds(params: PaginationParams, total_items: int) -> tuple[int, int]:
    """Calculate start and end indices for pagination.

    Args:
        params: page size etc.
        total_items: Total number of items available

    Returns:
        Tuple of (start_index, end_index) for slicing
    """
    start_idx = (params.page - 1) * params.page_size
    end_idx = min(start_idx + params.page_size, total_items)
    return start_idx, end_idx


def _filter_samples_by_class(
    samples: list[DataSample], class_filter: list[int], is_multilabel: bool
) -> list[DataSample]:
    """Filter samples by class labels.

    Args:
        samples: List of data samples to filter
        class_filter: List of class IDs to filter by
        is_multilabel: Whether the dataset is multilabel

    Returns:
        Filtered list of samples
    """
    if is_multilabel:
        # For multilabel: check if any of the filtered classes are active
        filtered_samples = []
        for sample in samples:
            if isinstance(sample.label, list):
                # Check if any of the filtered class indices have value 1
                if any(class_id < len(sample.label) and sample.label[class_id] == 1 for class_id in class_filter):
                    filtered_samples.append(sample)
            elif sample.label in class_filter:
                # Handle edge case where multilabel sample has single int label
                filtered_samples.append(sample)
        return filtered_samples

    return [s for s in samples if s.label in class_filter]


def main() -> None:
    """Main entry point for the MCP server."""
    settings = get_settings()
    if settings.transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=settings.transport, host=settings.host, port=settings.port)

