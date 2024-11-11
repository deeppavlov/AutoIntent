import json
import logging
from pathlib import Path

import hydra
import yaml

from autointent.configs.inference_cli import InferenceConfig

from .inference_pipeline import InferencePipeline


@hydra.main(config_name="inference_config", config_path=".", version_base=None)
def main(cfg: InferenceConfig) -> None:
    # load data for prediction
    with Path(cfg.data_path).open() as file:
        data: list[str] = json.load(file)

    # load pipeline config
    with (Path(cfg.source_dir) / "inference_config.yaml").open() as file:
        inference_config = yaml.safe_load(file)

    logger = logging.getLogger(__name__)
    logger.debug("Inference config loaded")

    # instantiate pipeline
    pipeline = InferencePipeline.from_dict_config(inference_config["nodes_configs"])

    # send data to pipeline
    output = pipeline.predict_with_metadata(data)

    # save results
    with Path(cfg.output_path).open("w") as file:
        json.dump(output.model_dump(exclude_none=True), file, indent=4)
