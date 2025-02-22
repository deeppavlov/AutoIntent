"""
Evolutionary strategy to augmenting utterances.

Deeply inspired by DeepEval evolutions.
"""

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from autointent import Dataset, Pipeline
from autointent.custom_types import Split
from autointent.generation.utterances.evolution.chat_templates import EvolutionChatTemplate
from autointent.generation.utterances.evolution.evolver import UtteranceEvolver
from autointent.generation.utterances.generator import Generator

SEARCH_SPACE = [
    {
        "node_type": "scoring",
        "target_metric": "scoring_roc_auc",
        "metrics": ["scoring_accuracy"],
        "search_space": [
            {
                "module_name": "linear",
                "embedder_config": ["sentence-transformers/all-MiniLM-L6-v2"],
            }
        ],
    },
    {
        "node_type": "decision",
        "target_metric": "decision_accuracy",
        "search_space": [
            {"module_name": "argmax"},
        ],
    },
]


class IncrementalUtteranceEvolver(UtteranceEvolver):
    """Incremental evolutionary strategy to augmenting utterances."""

    def __init__(
        self,
        generator: Generator,
        prompt_makers: Sequence[EvolutionChatTemplate],
        seed: int = 0,
        async_mode: bool = False,
        search_space: str | None = None,
    ) -> None:
        """Initialize."""
        super().__init__(generator, prompt_makers, seed, async_mode)
        self.search_space = self._choose_search_space(search_space)

    def _choose_search_space(self, search_space: str | None) -> list[dict[str, Any]] | Path | str:
        if search_space is None:
            return SEARCH_SPACE
        return search_space

    def augment(
        self,
        dataset: Dataset,
        split_name: str = Split.TRAIN,
        n_evolutions: int = 1,
        update_split: bool = True,
        batch_size: int = 4,
    ) -> HFDataset:
        """
        Augment some split of dataset.

        Note that for now it supports only single-label datasets.
        """
        best_result = 0
        merge_dataset = copy.deepcopy(dataset)
        generated_samples = []

        for _ in range(n_evolutions):
            new_samples_dataset = super().augment(
                dataset, split_name=split_name, n_evolutions=1, update_split=False, batch_size=batch_size
            )
            merge_dataset[split_name] = concatenate_datasets([merge_dataset[split_name], new_samples_dataset])
            generated_samples.append(new_samples_dataset)

            pipeline_optimizer = Pipeline.from_search_space(self.search_space)
            ctx = pipeline_optimizer.fit(merge_dataset)
            results = ctx.optimization_info.dump_evaluation_results()
            decision_metric = results["metrics"]["decision"][0]

            if decision_metric > best_result:
                best_result = decision_metric
            else:
                break

        if update_split:
            dataset[split_name] = merge_dataset[split_name]

        return concatenate_datasets(generated_samples)
