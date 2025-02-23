"""
Evolutionary strategy to augmenting utterances.

Deeply inspired by DeepEval evolutions.
"""
import copy
import random
from pathlib import Path
from typing import Any

import dspy
from datasets import Dataset as HFDataset, concatenate_datasets
from dspy.evaluate import SemanticF1

from autointent import Dataset, Pipeline
from autointent.custom_types import Split

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


# Define a DSPy signature for text augmentation.
class TextAugmentSignature(dspy.Signature):
    text: str = dspy.InputField()
    # n_examples: int = dspy.InputField()
    augmented_texts: list[str] = dspy.OutputField(
        desc="List of augmented texts that preserve the original meaning but use varied phrasing."
    )


# # Define a DSPy module that implements text augmentation.
# class TextAugmenter(dspy.Module):
#     def __init__(self) -> None:
#         # Here, we use a ChainOfThought module with the defined signature.
#         # The module is responsible for "thinking through" and generating multiple text variants.
#         super().__init__()
#         self.generator = dspy.ChainOfThought("text, n_examples -> augmented_texts")
#
#     def forward(self, text: str, n_examples: int) -> dspy.Prediction:
#         # Invoke the underlying generator with the input text and desired number of examples.
#         return self.generator(text=text, n_examples=n_examples)


class DSPYIncrementalUtteranceEvolver:
    """Incremental evolutionary strategy to augmenting utterances using DSPy."""

    def __init__(
        self,
        seed: int = 0,
        search_space: str | None = None,
    ) -> None:
        """Initialize."""
        self.search_space = self._choose_search_space(search_space)
        random.seed(seed)

        turbo = dspy.LM(
            'openai/model_name',
            api_base="http://...",
            api_key="test",
            model_type='text'
        )
        dspy.settings.configure(lm=turbo)
        # self.generator = dspy.ChainOfThought("text, n_examples -> augmented_texts: list[str]")
        self.generator = dspy.ChainOfThought("text -> augmented_texts: list[str]")

    def _choose_search_space(self, search_space: str | None) -> list[dict[str, Any]] | Path | str:
        if search_space is None:
            return SEARCH_SPACE
        return search_space

    def augment(
        self,
        dataset: Dataset,
        split_name: str = Split.TEST,
        n_evolutions: int = 1,
        update_split: bool = True,
        batch_size: int = 4,
    ) -> HFDataset:
        """
        Augment dataset split using DSPy with incremental optimization.
        """
        best_result = 0
        merge_dataset = copy.deepcopy(dataset)
        generated_samples = []
        original_split = dataset[split_name]

        dspy_dataset = [
            dspy.Example(
                text=sample[Dataset.utterance_feature],
                # n_examples=1,
                augmented_texts=sample[Dataset.utterance_feature]  # Use original as reference
            ).with_inputs(
                "text",
                # "n_examples"
            )
            for sample in original_split
        ]

        for _ in range(n_evolutions):
            # Optimize prompts using DSPy
            # evaluate = dspy.Evaluate(
            #     devset=dspy_dataset,
            #     metric=SemanticF1,
            #     num_threads=batch_size,
            #     display_progress=True,
            # )
            # optimizer = dspy.MIPROv2(
            #     metric=SemanticF1,
            #     auto="medium",
            #     num_threads=batch_size,
            #     log_dir="logs"
            # )
            # optimized_module = optimizer.compile(
            #     self.generator,
            #     trainset=dspy_dataset,
            #     requires_permission_to_run=False,
            #     max_bootstrapped_demos=4,
            #     max_labeled_demos=4
            # )
            # evaluate(optimized_module)

            # Generate new samples
            new_samples = []
            for sample in original_split:
                utterance = sample[Dataset.utterance_feature]
                label = sample[Dataset.label_feature]
                prediction = self.generator(text=utterance)
                new_samples.extend([{
                    Dataset.label_feature: label,
                    Dataset.utterance_feature: ut
                } for ut in prediction.augmented_texts])

            new_samples_dataset = HFDataset.from_list(new_samples)
            merge_dataset[split_name] = concatenate_datasets([merge_dataset[split_name], new_samples_dataset])
            generated_samples.append(new_samples_dataset)

            # Check if the new samples improve the model
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


if __name__ == "__main__":
    from autointent import Dataset

    # Example usage
    dataset = Dataset.from_hub("AutoIntent/clinc150_subset")
    evolver = DSPYIncrementalUtteranceEvolver(
        seed=42,
        search_space=None
    )
    augmented_dataset = evolver.augment(dataset, split_name=Split.TEST, n_evolutions=5)