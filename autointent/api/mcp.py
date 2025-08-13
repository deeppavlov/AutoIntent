"""MCP Server for AutoIntent."""

import logging
import math
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

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
    dataset_path: str = Field(..., description="Path to the training dataset JSON file")


class PredictInput(BaseModel):
    """Input model for the predict tool."""

    utterances: list[str] = Field(..., description="List of text utterances to classify")


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


class ClassesInput(BaseModel):
    """Input model for the classes tool."""

    pagination_params: PaginationParams


class ClassesOutput(BaseModel):
    """Output model for the classes tool."""

    classes: list[Intent] = Field(..., description="List of class information")
    pagination_info: PaginationInfo


class TrainDataInput(BaseModel):
    """Input model for the train_data tool."""

    class_filter: list[int] | None = Field(default=None, description="Filter by specific class IDs")
    pagination_params: PaginationParams


class DataSample(BaseModel):
    """Single training sample."""

    id: int
    text: str
    label: LabelWithOOS


class TrainDataOutput(BaseModel):
    """Output model for the train_data tool."""

    samples: list[DataSample] = Field(..., description="List of training samples")
    pagination_info: PaginationInfo


settings = Settings()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_pipeline() -> Pipeline:
    """Load the optimized pipeline from disk."""
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
    """Load the training dataset from disk."""
    dataset_path = Path(settings.dataset_path)
    if not dataset_path.exists():
        msg = f"Dataset path does not exist: {dataset_path}"
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

    Available tools:
    - predict: Classify text utterances using the trained model
    - classes: Get paginated list of all available classes/intents with their metadata
    - train_data: Access training samples with pagination and filtering options

    This server allows LLMs to leverage AutoIntent's powerful text classification models for text classification tasks.
    """,
    lifespan=lifespan,
)


@mcp.tool()
def predict(input_data: PredictInput) -> PredictOutput:
    """Classify text utterances using the trained AutoIntent pipeline.

    Args:
        input_data: Contains list of utterances to classify

    Returns:
        Predictions for each utterance
    """
    pipeline = load_pipeline()

    if not input_data.utterances:
        return PredictOutput(predictions=[])

    try:
        predictions = pipeline.predict(input_data.utterances)
        return PredictOutput(predictions=predictions)
    except Exception:
        logger.exception("Failed to make predictions")
        raise


@mcp.tool()
def classes(input_data: ClassesInput) -> ClassesOutput:
    """Get paginated list of all available classes/intents with their metadata.

    Args:
        input_data: Contains pagination parameters

    Returns:
        Paginated list of classes with metadata
    """
    dataset = load_dataset()

    total_classes = dataset.n_classes
    start_idx, end_idx = _calculate_page_bounds(
        page=input_data.pagination_params.page,
        page_size=input_data.pagination_params.page_size,
        total_items=total_classes,
    )

    # Get the paginated slice of intents
    paginated_intents = dataset.intents[start_idx:end_idx]

    pagination_info = PaginationInfo(
        total_items=total_classes, total_pages=math.ceil(total_classes / input_data.pagination_params.page_size)
    )

    return ClassesOutput(classes=paginated_intents, pagination_info=pagination_info)


@mcp.tool()
def train_data(input_data: TrainDataInput) -> TrainDataOutput:
    """Access training samples with pagination and filtering options.

    Args:
        input_data: Contains pagination and filtering parameters

    Returns:
        Paginated and filtered training samples
    """
    dataset = load_dataset()
    split_data = dataset[Split.TRAIN]

    # Convert to DataSample list
    samples_list: list[DataSample] = []
    for i in range(len(split_data)):
        sample = DataSample(
            id=i, text=split_data[i][dataset.utterance_feature], label=split_data[i][dataset.label_feature]
        )
        samples_list.append(sample)

    # Apply class filtering if specified
    if input_data.class_filter is not None:
        samples_list = _filter_samples_by_class(samples_list, input_data.class_filter, dataset.multilabel)

    total_samples = len(samples_list)
    start_idx, end_idx = _calculate_page_bounds(
        page=input_data.pagination_params.page,
        page_size=input_data.pagination_params.page_size,
        total_items=total_samples,
    )

    # Get paginated samples
    paginated_samples = samples_list[start_idx:end_idx]

    pagination_info = PaginationInfo(
        total_items=total_samples,
        total_pages=math.ceil(total_samples / input_data.pagination_params.page_size) if total_samples > 0 else 1,
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


# Make the server runnable
if __name__ == "__main__":
    mcp.run()
