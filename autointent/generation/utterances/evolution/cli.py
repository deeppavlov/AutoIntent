"""CLI for evolutionary augmenter."""

from argparse import ArgumentParser

from autointent import load_dataset
from autointent.generation.utterances.evolution.evolver import UtteranceEvolver
from autointent.generation.utterances.generator import Generator

from .chat_templates import AbstractEvolution, ConcreteEvolution, EvolutionChatTemplate, ReasoningEvolution


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
    parser.add_argument("--private", action="store_true", help="Publish privately if --output-repo option is used")
    parser.add_argument("--n-evolutions", type=int, default=1, help="Number of utterances to generate for each intent")
    parser.add_argument("--reasoning", action="store_true", help="Whether to use `Reasoning` evolution")
    parser.add_argument("--concretizing", action="store_true", help="Whether to use `Concretizing` evolution")
    parser.add_argument("--abstract", action="store_true", help="Whether to use `Abstract` evolution")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    evolutions: list[EvolutionChatTemplate] = []
    if args.reasoning:
        evolutions.append(ReasoningEvolution())
    if args.concretizing:
        evolutions.append(ConcreteEvolution())
    if args.abstract:
        evolutions.append(AbstractEvolution())

    dataset = load_dataset(args.input_path)

    generator = UtteranceEvolver(Generator(), evolutions, args.seed)
    generator.augment(dataset, n_evolutions=args.n_evolutions)

    dataset.to_json(args.output_path)

    if args.output_repo is not None:
        dataset.push_to_hub(args.output_repo)


if __name__ == "__main__":
    main()
