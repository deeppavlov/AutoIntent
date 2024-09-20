import json
import os
from argparse import ArgumentParser
from typing import Any

from ..generator import Generator
from .utterance_generator import LengthType, StyleType, UtteranceGenerator


def read_json_dataset(file_path: os.PathLike):
    with open(file_path) as file:
        return json.load(file)


def save_json_dataset(file_path: os.PathLike, intents: list[dict[str, Any]]):
    dirname = os.path.dirname(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(file_path, "w") as file:
        json.dump(intents, file, indent=4, ensure_ascii=False)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to json with intent records",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Where to save result",
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        required=True,
        help="Number of utterances to generate for each intent",
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
        choices=LengthType.__args__,
        default="none",
        help="How to extend the prompt with length instruction",
    )
    parser.add_argument(
        "--style",
        choices=StyleType.__args__,
        default="none",
        help="How to extend the prompt with style instruction",
    )
    parser.add_argument(
        "--same-punctuation",
        action="store_true",
        help="Whether to extend the prompt with punctuation instruction",
    )
    args = parser.parse_args()

    intents = read_json_dataset(args.input_path)

    generator = UtteranceGenerator(Generator(), args.custom_instruction, args.length, args.style, args.same_punctuation)
    for intent_record in intents:
        generator(intent_record, args.n_shots, inplace=True)

    save_json_dataset(args.output_path, intents)


if __name__ == "__main__":
    main()
