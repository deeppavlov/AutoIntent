"""Evolutionary strategy to augmenting utterances."""

import copy
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import dspy
except ImportError:
    import_error = "dspy is not installed. Please install it with `pip install dspy` or `pip install autointent[dspy]`."
    raise ImportError(import_error) from None

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets
from dspy.evaluate.auto_evaluation import f1_score

from autointent import Dataset, Pipeline
from autointent.custom_types import Split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SEARCH_SPACE = [
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
    """Calculate the average ROUGE-1 F1 score between pairs of texts in true_texts and augmented_texts.

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
    true_tokens = "".join(c for c in true_text.lower() if c.isalnum() or c.isspace()).split()
    aug_tokens = "".join(c for c in augmented_text.lower() if c.isalnum() or c.isspace()).split()
    if not true_tokens or not aug_tokens:
        return 0.0
    true_counts = Counter(true_tokens)
    aug_counts = Counter(aug_tokens)
    # Calculate the token overlap using the minimum count for common tokens
    overlap = sum(min(true_counts[token], aug_counts[token]) for token in true_counts.keys() & aug_counts.keys())
    precision = overlap / len(aug_tokens)
    recall = overlap / len(true_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


class SemanticRecallPrecision(dspy.Signature):  # type: ignore[misc]
    """Compare a system's response to the ground truth to compute its recall and precision.

    If asked to reason, enumerate key ideas in each response, and whether they are present in the other response.

    Copied from `dspy <https://github.com/stanfordnlp/dspy/blob/2957c5f998e0bc652017b6e3b1f8af34970b6f6b/dspy/evaluate/auto_evaluation.py#L4-L14>`_
    """

    question: str = dspy.InputField()
    ground_truth: str = dspy.InputField()
    system_response: str = dspy.InputField()
    recall: float = dspy.OutputField(desc="fraction (out of 1.0) of ground truth covered by the system response")
    precision: float = dspy.OutputField(desc="fraction (out of 1.0) of system response covered by the ground truth")


class AugmentSemanticF1(dspy.Module):  # type: ignore[misc]
    """Compare a system's response to the ground truth to compute its recall and precision.

    Adapted from `dspy SemanticF1 <https://dspy.ai/api/evaluation/SemanticF1/>_
    """

    def __init__(self, threshold: float = 0.66) -> None:
        """Initialize the AugmentSemanticF1.

        Args:
            threshold: Threshold for the boolean output.
        """
        self.threshold = threshold
        self.module = dspy.ChainOfThought(SemanticRecallPrecision)

    def forward(
        self, example: dspy.Example, pred: dspy.Prediction, trace: list[dspy.Prediction] | None = None
    ) -> float | bool:
        """Compute the score for the given example and prediction.

        Uses SemanticF1 as the base metric with a ROUGE-1 as repetition penalty.

        Args:
            example: Question and ground truth.
            pred: System response.
            trace: Predictions from previous iterations.

        Returns:
            The final score or a boolean based on the threshold.
        """
        # Compute base scores using the existing semantic metric.
        scores = self.module(
            question=example.text, ground_truth=example.augmented_text, system_response=pred.augmented_text
        )
        base_score = f1_score(scores.precision, scores.recall)

        # Compute repetition penalty factor.
        penalty = repetition_factor(example.augmented_text, pred.augmented_text)
        # length_penalty = len(example.augmented_text) / len(pred.augmented_text)
        # Apply penalty to the base score.
        final_score = base_score * penalty  # * length_penalty
        # Return the final score, or a boolean based on the threshold if trace is provided.
        return final_score if trace is None else final_score >= self.threshold  # type: ignore[no-any-return]


class AugmentationSignature(dspy.Signature):  # type: ignore[misc]
    """Signature for text generation for augmentation task."""

    text: str = dspy.InputField(desc="Text to augment. Your task to paraphrase this text.")
    augmented_text: str = dspy.OutputField(desc="Augmented text. This should be on same language as text")


class DSPYIncrementalUtteranceEvolver:
    """Incremental evolutionary strategy to augmenting utterances using DSPy.

    Implements an evolutionary strategy to augment utterances using DSPy. This module would augment the utterances.
    For ground truth utterances, it would generate new utterances and evaluate them using the pipeline.

    For scoring generations it would use modified SemanticF1 as the base metric with a ROUGE-1 as repetition penalty.

    See tutorial :ref:`evolutionary_strategy_augmentation` for usage examples.

    Args:
        model: Model name. This should follow naming schema from `litellm providers <https://docs.litellm.ai/docs/providers>`_.
        api_base: API base URL. Some models require this.
        temperature: Sampling temperature. 0.0 is default from dspy LM.
        max_tokens: Maximum number of tokens to generate. 1000 is default from dspy LM.
        seed: Random seed for reproducibility.
        search_space: Search space for the pipeline.

    """

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        seed: int = 42,
        search_space: str | None = None,
    ) -> None:
        """Initialize the DSPYIncrementalUtteranceEvolver."""
        self._search_space = search_space or DEFAULT_SEARCH_SPACE
        random.seed(seed)

        llm = dspy.LM(
            model,
            api_base=api_base,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        dspy.settings.configure(lm=llm)
        self._generator = dspy.ChainOfThoughtWithHint(AugmentationSignature)

    def augment(  # noqa: C901
        self,
        dataset: Dataset,
        split_name: str = Split.TEST,
        n_evolutions: int = 3,
        update_split: bool = True,
        mipro_init_params: dict[str, Any] | None = None,
        mipro_compile_params: dict[str, Any] | None = None,
        save_path: Path | str | None = None,
    ) -> HFDataset:
        """Augment the dataset using the evolutionary strategy.

        Args:
            dataset: The dataset to augment.
            split_name: The name of the split to augment.
            n_evolutions: Number of evolutions to perform.
            update_split: Whether to update the split with the augmented data.
            mipro_init_params: Parameters for the MIPROv2 augmentation.
                `Full list of parameters <https://dspy.ai/deep-dive/optimizers/miprov2/#initialization-parameters>`_
            mipro_compile_params: Parameters for the MIPROv2 compilation.
                `Full list of params available <https://dspy.ai/deep-dive/optimizers/miprov2/#compile-parameters>`_
            save_path: Path to save the prompt of LLM. If None is provided, it will not be saved.

        Returns:
            The augmented dataset.
        """
        best_result = 0
        merge_dataset = copy.deepcopy(dataset)
        generated_samples = []
        original_split = dataset[split_name]
        if mipro_init_params is None:
            mipro_init_params = {}
        if mipro_compile_params is None:
            mipro_compile_params = {}

        if save_path is not None:
            if isinstance(save_path, str):
                save_path = Path(save_path)

            if not save_path.exists():
                save_path.mkdir(parents=True)

        dspy_dataset = [
            dspy.Example(
                text=sample[Dataset.utterance_feature],
                augmented_text=sample[Dataset.utterance_feature],  # Use original as reference
            ).with_inputs(
                "text",
            )
            for sample in original_split
        ]

        for i in range(n_evolutions):
            metric = AugmentSemanticF1()

            optimizer = dspy.MIPROv2(metric=metric, **mipro_init_params)

            optimized_module = optimizer.compile(self._generator, trainset=dspy_dataset, **mipro_compile_params)

            if save_path is not None:
                optimized_module.save((save_path / f"evolution_{i}").as_posix(), save_program=True)
                optimized_module.save(
                    (save_path / f"evolution_{i}" / "generator_state.json").as_posix(), save_program=False
                )
            # Generate new samples
            new_samples = []
            for sample in original_split:
                utterance = sample[Dataset.utterance_feature]
                label = sample[Dataset.label_feature]
                prediction = optimized_module(text=utterance)
                new_samples.append({Dataset.label_feature: label, Dataset.utterance_feature: prediction.augmented_text})

            new_samples_dataset = HFDataset.from_list(new_samples)
            merge_dataset[split_name] = concatenate_datasets([merge_dataset[split_name], new_samples_dataset])
            generated_samples.append(new_samples_dataset)

            # Check if the new samples improve the model
            pipeline_optimizer = Pipeline.from_search_space(self._search_space)
            ctx = pipeline_optimizer.fit(merge_dataset)
            results = ctx.optimization_info.dump_evaluation_results()
            decision_metric = results["metrics"]["decision"][0]
            msg = f"Evolution {i} decision metric: {decision_metric}"
            logger.info(msg)

            if decision_metric > best_result:
                best_result = decision_metric
                msg = f"Evolution {i} is the best so far."
                logger.info(msg)
            else:
                break

        if update_split:
            dataset[split_name] = merge_dataset[split_name]

        return concatenate_datasets(generated_samples)
