import json
from pathlib import Path

from autointent import OptimizationConfig


def generate_json_schema_optimizer_config() -> None:
    """Generate the JSON schema for the optimizer config."""
    schema = OptimizationConfig.model_json_schema()
    path = Path(__file__).parent.parent / "docs" / "optimizer_config.schema.json"
    with path.open("w") as f:
        json.dump(schema, f, indent=4)


if __name__ == "__main__":
    generate_json_schema_optimizer_config()
