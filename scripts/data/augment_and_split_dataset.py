import json
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import List
from collections import defaultdict
from autointent import Dataset
from autointent.generation.utterances.basic.chat_template import SynthesizerChatTemplateRussian
from autointent.generation.utterances.basic.utterance_generator import UtteranceGenerator
from autointent.generation.utterances.generator import Generator

def process_utterances(generated: List[str]) -> List[str]:
    processed = []
    for ut in generated:
        if "', '" in ut or "',\n" in ut:
            clean_ut = ut.replace("[", "").replace("]", "").replace("'", "")
            split_ut = [u.strip() for u in clean_ut.split(", ") if u.strip()]
            processed.extend(split_ut)
        else:
            processed.append(ut.strip())
    return processed

def main():
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to few-shot dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save generated datasets")
    parser.add_argument("--n-augment", type=int, required=True, 
                      help="Max number of augmented examples per class to generate")
    parser.add_argument("--split-numbers", type=int, nargs="+", required=True,
                      help="List of example counts to split into (e.g. 1 2 3 5)")
    parser.add_argument("--max-attempts", type=int, default=5,
                      help="Max generation attempts per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_json(args.input_path)
    template = SynthesizerChatTemplateRussian(dataset, split="train")
    generator = UtteranceGenerator(Generator(), template)

    augmented_samples = []
    for intent in dataset.intents:
        print(f"\nProcessing intent: {intent.name} (ID: {intent.id})")
        
        valid_utterances = []
        attempts = 0
        
        while len(valid_utterances) < args.n_augment and attempts < args.max_attempts:
            needed = args.n_augment - len(valid_utterances)
            generated = generator(intent_data=intent, n_generations=needed)
            
            processed = process_utterances(generated)
            current_valid = [
                ut for ut in processed 
                if ut and len(ut.split()) > 2
            ]
            valid_utterances.extend(current_valid)
            
            print(f"Attempt {attempts+1}: "
                  f"Generated {len(current_valid)} valid, "
                  f"Total {len(valid_utterances)}/{args.n_augment}")
            attempts += 1

        if len(valid_utterances) < args.n_augment:
            raise RuntimeError(
                f"Failed to generate {args.n_augment} examples for "
                f"{intent.name} after {args.max_attempts} attempts"
            )
            
        augmented_samples.extend([
            {"utterance": ut, "label": intent.id} 
            for ut in valid_utterances[:args.n_augment]
        ])

    raw_augmented_path = Path(args.output_dir) / "raw_augmented_samples.json"
    with open(raw_augmented_path, "w", encoding="utf-8") as f:
        json.dump({
            "intents": [{"id": intent.id, "name": intent.name} for intent in dataset.intents],
            "samples": augmented_samples
        }, f, indent=4, ensure_ascii=False)

    splits = {}
    max_num = max(args.split_numbers)
    for n in args.split_numbers:
        if n > max_num:
            raise ValueError(f"Requested {n} examples but max is {max_num}")
        
        class_to_samples = defaultdict(list)
        for sample in augmented_samples:
            class_to_samples[sample["label"]].append(sample)
        
        selected = []
        for class_id, samples in class_to_samples.items():
            selected.extend(random.sample(samples, k=n))
        
        splits[n] = selected

    original_data = dataset["train"].to_list()
    for n, aug_samples in splits.items():
        combined = original_data + aug_samples
        
        new_dataset = Dataset.from_dict({
            "intents": dataset.intents,
            "train": combined
        })
        
        output_path = Path(args.output_dir) / f"dataset_{n}_examples.json"
        new_dataset.to_json(output_path)
        print(f"Saved {len(combined)} examples to {output_path}")

if __name__ == "__main__":
    main()