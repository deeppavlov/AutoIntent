"""Tests for the AutoIntent MCP Server."""

import os
import tempfile
from pathlib import Path

import pytest

from autointent import Pipeline
from autointent.configs import LoggingConfig
from tests.conftest import get_search_space

# Skip all tests in this file if FastMCP dependencies are not available
fastmcp = pytest.importorskip("fastmcp")
Client = pytest.importorskip("fastmcp").Client


@pytest.fixture
def trained_pipeline_path(dataset):
    """Train a pipeline and save it to a temporary directory along with the dataset."""
    # Create temporary directory for this test
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / "test_mcp"

        # Get a light search space for faster testing
        search_space = get_search_space("light")

        # Create and train pipeline
        pipeline_optimizer = Pipeline.from_search_space(search_space)
        logging_config = LoggingConfig(project_dir=project_dir, dump_modules=True, clear_ram=True)
        pipeline_optimizer.set_config(logging_config)

        # Train and save pipeline (this automatically saves dataset.json in the same directory)
        context = pipeline_optimizer.fit(dataset)
        context.dump()

        yield logging_config.dirpath


@pytest.fixture
def mcp_server(trained_pipeline_path):
    """Create an MCP server with the trained pipeline."""
    # Set environment variables for the MCP server
    original_path = os.environ.get("AUTOINTENT_PATH")

    os.environ["AUTOINTENT_PATH"] = str(trained_pipeline_path)

    try:
        # Import after setting environment variables to ensure they're picked up
        from autointent.server.mcp import mcp

        yield mcp
    finally:
        # Restore original environment
        if original_path is not None:
            os.environ["AUTOINTENT_PATH"] = original_path
        else:
            os.environ.pop("AUTOINTENT_PATH", None)


@pytest.mark.asyncio
async def test_mcp_server_tools_list(mcp_server):
    """Test that the MCP server exposes the expected tools."""
    async with Client(mcp_server) as client:
        tools = await client.list_tools()
        tool_names = {tool.name for tool in tools}

        expected_tools = {"predict", "classes", "train_data"}
        assert tool_names == expected_tools


@pytest.mark.asyncio
async def test_predict_tool_single_utterance(mcp_server):
    """Test the predict tool with a single utterance."""
    async with Client(mcp_server) as client:
        result = await client.call_tool("predict", {"utterances": ["hello world"]})

        # Extract the actual data from the CallToolResult
        data = result.data

        assert hasattr(data, "predictions")
        assert len(data.predictions) == 1
        # Single-label classification - can be int (in-scope) or None (out-of-scope)
        assert data.predictions[0] is None or isinstance(data.predictions[0], int)


@pytest.mark.asyncio
async def test_predict_tool_multiple_utterances(mcp_server):
    """Test the predict tool with multiple utterances."""
    test_utterances = ["hello world", "book a flight", "what's the weather like"]

    async with Client(mcp_server) as client:
        result = await client.call_tool("predict", {"utterances": test_utterances})

        # Extract the actual data from the CallToolResult
        data = result.data

        assert hasattr(data, "predictions")
        assert len(data.predictions) == len(test_utterances)

        # Check that all predictions are integers (in-scope) or None (out-of-scope)
        for prediction in data.predictions:
            assert prediction is None or isinstance(prediction, int)


@pytest.mark.asyncio
async def test_predict_tool_empty_utterances(mcp_server):
    """Test the predict tool with empty utterances list."""
    async with Client(mcp_server) as client:
        result = await client.call_tool("predict", {"utterances": []})

        # Extract the actual data from the CallToolResult
        data = result.data

        assert hasattr(data, "predictions")
        assert data.predictions == []


@pytest.mark.asyncio
async def test_classes_tool_default_pagination(mcp_server):
    """Test the classes tool with default pagination."""
    async with Client(mcp_server) as client:
        result = await client.call_tool("classes", {"page": 1, "page_size": 3})

        # Extract the actual data from the CallToolResult
        data = result.data

        assert hasattr(data, "classes")
        assert hasattr(data, "pagination_info")
        assert isinstance(data.classes, list)
        assert len(data.classes) <= 3  # Should not exceed page size

        pagination = data.pagination_info
        assert hasattr(pagination, "total_items")
        assert hasattr(pagination, "total_pages")
        assert pagination.total_items >= 0
        assert pagination.total_pages >= 1


@pytest.mark.asyncio
async def test_classes_tool_custom_pagination(mcp_server):
    """Test the classes tool with custom pagination parameters."""
    async with Client(mcp_server) as client:
        result = await client.call_tool("classes", {"page": 1, "page_size": 2})

        # Extract the actual data from the CallToolResult
        data = result.data

        assert hasattr(data, "classes")
        assert len(data.classes) <= 2  # Should not exceed requested page size

        # Each class should have the expected structure
        for class_info in data.classes:
            assert hasattr(class_info, "id")
            assert hasattr(class_info, "name")


@pytest.mark.asyncio
async def test_train_data_tool_default_pagination(mcp_server):
    """Test the train_data tool with default pagination."""
    async with Client(mcp_server) as client:
        result = await client.call_tool("train_data", {"page": 1, "page_size": 3})

        # Extract the actual data from the CallToolResult
        data = result.data

        assert hasattr(data, "samples")
        assert hasattr(data, "pagination_info")
        assert isinstance(data.samples, list)
        assert len(data.samples) <= 3  # Should not exceed page size

        pagination = data.pagination_info
        assert hasattr(pagination, "total_items")
        assert hasattr(pagination, "total_pages")
        assert pagination.total_items >= 0

        # Each sample should have the expected structure
        for sample in data.samples:
            assert hasattr(sample, "id")
            assert hasattr(sample, "text")
            assert hasattr(sample, "label")
            assert isinstance(sample.id, int)
            assert isinstance(sample.text, str)


@pytest.mark.asyncio
async def test_train_data_tool_with_class_filter(mcp_server):
    """Test the train_data tool with class filtering."""
    async with Client(mcp_server) as client:
        # First get some classes to use as filter
        classes_result = await client.call_tool("classes", {"page": 1, "page_size": 5})

        # Extract classes data
        classes_data = classes_result.data

        # Use the first available class ID for filtering
        if classes_data.classes:
            class_id = classes_data.classes[0].id

            result = await client.call_tool("train_data", {"class_filter": [class_id], "page": 1, "page_size": 10})

            # Extract the actual data from the CallToolResult
            data = result.data
            assert hasattr(data, "samples")
            assert hasattr(data, "pagination_info")

            # All samples should match the filtered class
            for sample in data.samples:
                # For single-label classification, the label should match the filter
                # For multi-label, we need to check if the class is active
                if isinstance(sample.label, list):
                    # Multi-label case: check if the class_id index is 1
                    if class_id < len(sample.label):
                        assert sample.label[class_id] == 1
                else:
                    # Single-label case: label should equal class_id
                    assert sample.label == class_id


@pytest.mark.asyncio
async def test_train_data_tool_empty_class_filter(mcp_server):
    """Test the train_data tool with an empty class filter."""
    async with Client(mcp_server) as client:
        result = await client.call_tool("train_data", {"class_filter": [], "page": 1, "page_size": 10})

        # Extract the actual data from the CallToolResult
        data = result.data

        assert hasattr(data, "samples")
        # With empty filter, no samples should be returned
        assert len(data.samples) == 0


@pytest.mark.asyncio
async def test_train_data_tool_nonexistent_class_filter(mcp_server):
    """Test the train_data tool with a non-existent class filter."""
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "train_data",
            {
                "class_filter": [9999],  # Assuming this class doesn't exist
                "page": 1,
                "page_size": 10,
            },
        )

        # Extract the actual data from the CallToolResult
        data = result.data

        assert hasattr(data, "samples")
        # With non-existent class filter, no samples should be returned
        assert len(data.samples) == 0


@pytest.mark.asyncio
async def test_server_instructions(mcp_server):
    """Test that the server has proper instructions."""
    async with Client(mcp_server) as client:  # noqa: F841
        # Get server info should include instructions
        # Note: This assumes the Client provides a way to get server info
        # If not available, we can at least verify the server name
        assert mcp_server.name == "AutoIntent MCP Server"
        assert "AutoIntent MCP Server provides text classification capabilities" in mcp_server.instructions
