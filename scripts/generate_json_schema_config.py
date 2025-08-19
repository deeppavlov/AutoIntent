import json
import logging
from pathlib import Path

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)

from autointent import OptimizationConfig
from autointent.schemas.node_validation import OptimizationSearchSpaceConfig


def generate_json_schema_optimizer_config() -> None:
    """Generate the JSON schema for the optimizer config."""
    logger.debug(f"Generating json schema for OptimizationConfig...")
    schema = OptimizationConfig.model_json_schema()
    path = Path(__file__).parent.parent / "docs" / "optimizer_config.schema.json"
    with path.open("w") as f:
        json.dump(schema, f, indent=4)

    logger.debug(f"Generating json schema for OptimizationSearchSpaceConfig...")
    schema = OptimizationSearchSpaceConfig.model_json_schema()
    path = Path(__file__).parent.parent / "docs" / "optimizer_search_space_config.schema.json"
    with path.open("w") as f:
        json.dump(schema, f, indent=4)


if __name__ == "__main__":
    generate_json_schema_optimizer_config()
