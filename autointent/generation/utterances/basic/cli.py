"""CLI for basic utterance generator."""

from argparse import ArgumentParser

from autointent import load_dataset
from autointent.generation.utterances.basic.utterance_generator import UtteranceGenerator
from autointent.generation.utterances.generator import Generator

from .chat_template import SynthesizerChatTemplate


def main() -> None:
    """ClI endpoint."""
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
    args = parser.parse_args()

    dataset = load_dataset(args.input_path)
    template = SynthesizerChatTemplate(dataset, "train", max_sample_utterances=args.n_sample_utterances)
    generator = UtteranceGenerator(Generator(), template)
    generator.augment(dataset, n_generations=args.n_generations)

    dataset.to_json(args.output_path)

    if args.output_repo is not None:
        dataset.push_to_hub(args.output_repo, private=args.private)


if __name__ == "__main__":
    main()
