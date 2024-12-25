from argparse import ArgumentParser
from time import sleep

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from autointent.generation.utterances.evolution.evolver import UtteranceEvolver
from autointent.generation.utterances.generator import Generator


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to json with intent records")
    parser.add_argument("--output-path", type=str, required=True, help="Where to save result")
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

    evolutions = []
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

    dataset_utterances = load_dataset(args.input_path)["train"]
    dataset_intents = load_dataset(args.input_path, name="intents")
    splitted_dataset_intents = dataset_intents["intents"]

    df_utterances = pd.DataFrame(dataset_utterances)
    df_intents = pd.DataFrame(splitted_dataset_intents)
    print(df_utterances.columns)
    print(df_intents.columns)

    df_merged = df_utterances.merge(df_intents[["id", "name"]], left_on="label", right_on="id", how="left")

    print(evolutions)
    generator = UtteranceEvolver(Generator(), evolutions, args.seed)
    new_utterances = []
    new_intents = []
    labels = []
    for idx, sample in df_merged.iterrows():
        print(sample)
        try:
            sleep(1)
            generated_utterances = generator(sample["utterance"].strip(), sample["name"], args.n_evolutions)
        except Exception as e:
            print(e)
            continue
        new_utterances.extend(generated_utterances)
        new_intents.extend([sample["name"]] * len(generated_utterances))
        labels.extend([sample["label"]] * len(generated_utterances))

    df_new_utterances = pd.DataFrame({"utterance": new_utterances, "label": labels})
    df_utterances = Dataset.from_pandas(pd.concat([df_utterances, df_new_utterances], ignore_index=True))

    dataset_dict = DatasetDict(
        {
            "train": df_utterances,
        }
    )
    dataset_dict.push_to_hub(args.output_path, "default")
    dataset_intents.push_to_hub(args.output_path, "intents")


if __name__ == "__main__":
    main()
