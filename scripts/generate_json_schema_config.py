import json
from pathlib import Path

from autointent.nodes.schemes import OptimizationSearchSpaceConfig
from autointent import OptimizationConfig


def generate_json_schema_search_space_config() -> None:
    """Generate the JSON schema for the optimizer config."""
    schema = OptimizationSearchSpaceConfig.model_json_schema()
    path = Path(__file__).parent.parent / "docs" / "optimizer_search_space_config.schema.json"
    with path.open("w") as f:
        json.dump(schema, f, indent=4)


def generate_json_schema_optimizer_config() -> None:
    """Generate the JSON schema for the optimizer config."""
    schema = OptimizationConfig.model_json_schema()
    path = Path(__file__).parent.parent / "docs" / "optimizer_config.schema.json"
    with path.open("w") as f:
        json.dump(schema, f, indent=4)


if __name__ == "__main__":
    generate_json_schema_search_space_config()
    generate_json_schema_optimizer_config()
