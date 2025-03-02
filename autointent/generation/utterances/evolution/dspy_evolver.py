"""
Evolutionary strategy to augmenting utterances.
"""

import copy
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import dspy
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

# from dspy.evaluate import CompleteAndGrounded, SemanticF1, answer_exact_match
from dspy.evaluate.auto_evaluation import f1_score

from autointent import Dataset, Pipeline
from autointent.custom_types import Split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def repetition_factor(true_text: str, augmented_text: str) -> float:
    """
    Calculate the average ROUGE-1 F1 score between pairs of texts in true_texts and augmented_texts.

    ROUGE-1 F1 is computed as:
        F1 = 2 * (precision * recall) / (precision + recall)
    where:
        - precision = (overlap in unigrams) / (total unigrams in augmented text)
        - recall = (overlap in unigrams) / (total unigrams in true text)

    Args:
        true_text: A ground truth text.
        augmented_text: A list of augmented/generated text.

    Returns:
        float: The average ROUGE-1 F1 score across all pairs.

    Raises:
        ValueError: If the lengths of true_texts and augmented_texts differ.
    """
    true_tokens = true_text.split()
    aug_tokens = augmented_text.split()
    if not true_tokens or not aug_tokens:
        return 0.0
    true_counts = Counter(true_tokens)
    aug_counts = Counter(aug_tokens)
    # Calculate the token overlap using the minimum count for common tokens
    overlap = sum(min(true_counts[token], aug_counts[token]) for token in true_counts.keys() & aug_counts.keys())
    precision = overlap / len(aug_tokens)
    recall = overlap / len(true_tokens)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1


class SemanticRecallPrecision(dspy.Signature):
    """
    Compare a system's response to the ground truth to compute its recall and precision.
    If asked to reason, enumerate key ideas in each response, and whether they are present in the other response.
    """

    # Copied from dspy

    question: str = dspy.InputField()
    ground_truth: str = dspy.InputField()
    system_response: str = dspy.InputField()
    recall: float = dspy.OutputField(desc="fraction (out of 1.0) of ground truth covered by the system response")
    precision: float = dspy.OutputField(desc="fraction (out of 1.0) of system response covered by the ground truth")


class AugmentSemanticF1(dspy.Module):
    # adapted SemanticF1
    def __init__(self, threshold: float = 0.66, **kwargs: Any) -> None:
        self.threshold = threshold
        self.module = dspy.ChainOfThought(SemanticRecallPrecision)

    def forward(
        self, example: dspy.Example, pred: dspy.Prediction, trace: list[dspy.Prediction] | None = None
    ) -> float | bool:
        # Compute base scores using the existing semantic metric.
        scores = self.module(question=example.question, ground_truth=example.response, system_response=pred.response)
        base_score = f1_score(scores.precision, scores.recall)

        # Compute repetition penalty factor.
        penalty = repetition_factor(example.response, pred.response)

        # Apply penalty to the base score.
        final_score = base_score * penalty
        # Return the final score, or a boolean based on the threshold if trace is provided.
        return final_score if trace is None else final_score >= self.threshold


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
            "hosted_vllm/x5-airun-medium-coder-prod",
            api_base="http://mn-rtx01.x5.ru:8000/v1",
            # api_key="test",
            model_type="chat",
        )
        dspy.settings.configure(lm=turbo)
        # self.generator = dspy.ChainOfThought("text, n_examples -> augmented_texts: list[str]")
        # input should be question and response is augmented. question and response required for metric
        self.generator = dspy.ChainOfThought("question -> response: str")

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
                question=sample[Dataset.utterance_feature],
                # n_examples=1,
                response=sample[Dataset.utterance_feature],  # Use original as reference
            ).with_inputs(
                "question",
                # "n_examples"
            )
            for sample in original_split
        ]

        for i in range(n_evolutions):
            metric = AugmentSemanticF1()

            optimizer = dspy.MIPROv2(
                metric=metric,  # SemanticF1
                # auto="medium",  # can be low, medium, high. this setting will override params in compile
                # num_threads=batch_size,
                # log_dir="logs",
            )
            optimized_module = optimizer.compile(
                self.generator,
                trainset=dspy_dataset,
                requires_permission_to_run=False,
                minibatch=False,
                # max_bootstrapped_demos=4,
                # max_labeled_demos=4,
                num_trials=5,
            )
            # evaluate(optimized_module)
            # try:
            self.generator.save("generator/", save_program=True)
            # should be dir + file *.json or *.pkl
            self.generator.save("generator/generator_state.json", save_program=False)

            optimized_module.save("optimized_module", save_program=True)
            optimized_module.save("optimized_module/optimized_module.json", save_program=False)
            # Generate new samples
            new_samples = []
            for sample in original_split:
                utterance = sample[Dataset.utterance_feature]
                label = sample[Dataset.label_feature]
                prediction = optimized_module(question=utterance)
                new_samples.extend(
                    [{Dataset.label_feature: label, Dataset.utterance_feature: ut} for ut in prediction.response]
                )

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
    evolver = DSPYIncrementalUtteranceEvolver(seed=42, search_space=None)
    augmented_dataset = evolver.augment(dataset, split_name=Split.TEST, n_evolutions=2)
