"""Basic generation of new utterances from existing ones."""

from collections.abc import Callable

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from autointent import Dataset
from autointent.custom_types import Split
from autointent.generation.utterances.generator import Generator
from autointent.generation.utterances.schemas import Message
from autointent.schemas import Intent, Sample


class UtteranceGenerator:
    """
    Basic generation of new utterances from existing ones.

    This augmentation method simply prompts LLM to look at existing examples
    and generate similar. Additionaly it can consider some aspects of style,
    punctuation and length of the desired generations.
    """

    def __init__(self, generator: Generator, prompt_maker: Callable[[Intent, int], list[Message]]) -> None:
        """Initialize."""
        self.generator = generator
        self.prompt_maker = prompt_maker

    def __call__(self, intent_data: Intent, n_generations: int) -> list[str]:
        """Generate new utterances."""
        messages = self.prompt_maker(intent_data, n_generations)
        response_text = self.generator.get_chat_completion(messages)
        return _extract_utterances(response_text)

    def augment(
        self,
        dataset: Dataset,
        split_name: str = Split.TRAIN,
        n_generations: int = 5,
        update_split: bool = True,
    ) -> list[Sample]:
        """
        Augment some split of dataset.

        TODO Note that for now it supports only single-label datasets.
        """
        original_split = dataset[split_name]
        new_samples = []
        for intent in dataset.intents:
            generated_utterances = self(
                intent_data=intent,
                n_generations=n_generations,
            )
            new_samples.extend(
                [{Dataset.label_feature: intent.id, Dataset.utterance_feature: ut} for ut in generated_utterances]
            )
        if update_split:
            generated_split = HFDataset.from_list(new_samples)
            dataset[split_name] = concatenate_datasets([original_split, generated_split])
        return [Sample(**sample) for sample in new_samples]


def _extract_utterances(response_text: str) -> list[str]:
    """
    Parse LLM output.

    Inverse function to :py:func:`_format_utterances`.
    """
    raw_utterances = response_text.split("\n")
    # remove enumeration
    return [ut[ut.find(" ") + 1 :] for ut in raw_utterances]
