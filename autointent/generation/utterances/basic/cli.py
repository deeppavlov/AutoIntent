"""CLI for basic utterance generator."""

import logging
from argparse import ArgumentParser

from autointent import load_dataset
from autointent.generation import Generator
from autointent.generation.chat_templates import EnglishSynthesizerTemplate, RussianSynthesizerTemplate
from autointent.generation.utterances import UtteranceGenerator

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def main() -> None:
    """CLI endpoint."""
    parser = ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to json or hugging face repo with dataset",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Local path where to save result",
    )
    parser.add_argument(
        "--output-repo",
        type=str,
        default=None,
        help="Local path where to save result",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--private", action="store_true", help="Publish privately if --output-repo option is used")
    parser.add_argument(
        "--n-generations",
        type=int,
        default=5,
        help="Number of utterances to generate for each intent",
    )
    parser.add_argument(
        "--n-sample-utterances",
        type=int,
        default=5,
        help="Number of utterances to use as an example for augmentation",
    )
    parser.add_argument("--async-mode", action="store_true", help="Enable asynchronous generation")
    parser.add_argument("--language", choices=["ru", "en"], default="en")
    args = parser.parse_args()

    dataset = load_dataset(args.input_path)
    template_type = EnglishSynthesizerTemplate if args.language == "en" else RussianSynthesizerTemplate
    template = template_type(dataset, args.split, max_sample_utterances=args.n_sample_utterances)
    generator = UtteranceGenerator(Generator(), template, async_mode=args.async_mode)

    n_before = len(dataset[args.split])
    new_samples = generator.augment(dataset, split_name=args.split, n_generations=args.n_generations)
    n_after = len(dataset[args.split])

    logger.info("# samples before %s", n_before)
    logger.info("# samples generated %s", len(new_samples))
    logger.info("# samples after %s", n_after)

    dataset.to_json(args.output_path)

    if args.output_repo is not None:
        dataset.push_to_hub(args.output_repo, private=args.private)
