import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from random import seed, sample
from autointent import Dataset

def main() -> None:
    parser = ArgumentParser(description="Create few-shot version of multiclass dataset")
    parser.add_argument("--dataset-name", type=str, required=True, 
                      help="Hugging Face dataset path (e.g. 'AutoIntent/massive_ru')")
    parser.add_argument("--output-path", type=str, required=True,
                      help="Path to save few-shot dataset")
    parser.add_argument("--k-shots", type=int, required=True,
                      help="Number of examples per class")
    parser.add_argument("--split", type=str, default="train",
                      help="Dataset split to process")
    parser.add_argument("--seed", type=int, default=0,
                      help="Random seed for reproducibility")
    args = parser.parse_args()

    seed(args.seed)

    dataset = Dataset.from_hub(args.dataset_name)
    
    class_to_examples = defaultdict(list)
    for example in dataset[args.split]:
        class_to_examples[example["label"]].append(example["utterance"])

    fewshot_examples = []
    for class_id, utterances in class_to_examples.items():
        if len(utterances) < args.k_shots:
            raise ValueError(f"Class {class_id} has only {len(utterances)} examples")
        
        selected = sample(utterances, args.k_shots)
        fewshot_examples.extend([
            {"utterance": utt, "label": class_id} for utt in selected
        ])

    fewshot_dataset = Dataset.from_dict({
        "intents": dataset.intents,
        args.split: fewshot_examples
    })

    fewshot_dataset.to_json(args.output_path)

if __name__ == "__main__":
    main()