"""CLI for basic utterance generator."""

from argparse import ArgumentParser

from autointent import load_dataset
from autointent.generation.utterances.basic.utterance_generator import LengthType, StyleType, UtteranceGenerator
from autointent.generation.utterances.generator import Generator


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
    parser.add_argument(
        "--custom-instruction",
        type=str,
        action="append",
        help="Add extra instructions to default prompt."
        "You can use this argument multiple times to add multiple instructions",
    )
    parser.add_argument(
        "--length",
        choices=LengthType.__args__,  # type: ignore[attr-defined]
        default="none",
        help="How to extend the prompt with length instruction",
    )
    parser.add_argument(
        "--style",
        choices=StyleType.__args__,  # type: ignore[attr-defined]
        default="none",
        help="How to extend the prompt with style instruction",
    )
    parser.add_argument(
        "--same-punctuation",
        action="store_true",
        help="Whether to extend the prompt with punctuation instruction",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.input_path)
    generator = UtteranceGenerator(
        Generator(), args.custom_instruction or [], args.length, args.style, args.same_punctuation
    )
    generator.augment(dataset, n_generations=args.n_generations, max_sample_utterances=args.n_sample_utterances)

    dataset.to_json(args.output_path)

    if args.output_repo is not None:
        dataset.push_to_hub(args.output_repo, private=args.private)


if __name__ == "__main__":
    main()
