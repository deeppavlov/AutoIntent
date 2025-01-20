"""CLI for evolutionary augmenter."""

from argparse import ArgumentParser
from typing import TYPE_CHECKING

from autointent import load_dataset
from autointent.generation.utterances.evolution.evolver import UtteranceEvolver
from autointent.generation.utterances.generator import Generator

if TYPE_CHECKING:
    from .evolver import EvolutionType


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
    parser.add_argument("--formal", action="store_true", help="Whether to use `Formal` evolution")
    parser.add_argument("--informal", action="store_true", help="Whether to use `Informal` evolution")
    parser.add_argument("--funny", action="store_true", help="Whether to use `Funny` evolution")
    parser.add_argument("--goofy", action="store_true", help="Whether to use `Goofy` evolution")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    evolutions: list[EvolutionType] = []
    if args.reasoning:
        evolutions.append("reasoning")
    if args.concretizing:
        evolutions.append("concretizing")
    if args.abstract:
        evolutions.append("abstract")
    if args.formal:
        evolutions.append("formal")
    if args.informal:
        evolutions.append("informal")
    if args.funny:
        evolutions.append("funny")
    if args.goofy:
        evolutions.append("goofy")

    dataset = load_dataset(args.input_path)

    generator = UtteranceEvolver(Generator(), evolutions, args.seed)
    generator.augment(dataset, n_evolutions=args.n_evolutions)

    dataset.to_json(args.output_path)

    if args.output_repo is not None:
        dataset.push_to_hub(args.output_repo)


if __name__ == "__main__":
    main()
