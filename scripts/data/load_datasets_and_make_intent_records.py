import random

import nltk
from nltk.collocations import (
    BigramAssocMeasures,
    BigramCollocationFinder,
    TrigramAssocMeasures,
    TrigramCollocationFinder,
)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def get_stopwords(language):
    return set(stopwords.words(language))


def tokenize(text, language):
    # import spacy
    # if language == "english":
    #     nlp = spacy.load("en_core_web_sm")
    # elif language == "russian":
    #     nlp = spacy.load("ru_core_news_sm")

    # doc = nlp(text)

    # tokens = [token.text for token in doc]
    return word_tokenize(text, language)



def preprocess(text, language):
    tokens = tokenize(text, language=language)
    tokens = ["".join([c.lower() for c in word if c.isalnum()]) for word in tokens]
    return [word for word in tokens if len(word) > 0]


def find_ngrams(text: str, language, n_unigrams=1, n_bigrams=1, n_trigrams=1, stopwords_set=None):
    if stopwords_set is None:
        stopwords_set = set()
    tokens = preprocess(text, language=language)

    # from collections import Counter
    # cnt = Counter(filter(lambda x: x not in stopwords_set, tokens))
    # unigrams_raw = cnt.most_common(n_unigrams)
    # unigrams = [token for token, freq in unigrams_raw]

    bigram_measures = BigramAssocMeasures()
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(bigram_measures.student_t, n_bigrams)

    trigram_measures = TrigramAssocMeasures()
    trigram_finder = TrigramCollocationFinder.from_words(tokens)
    trigrams = trigram_finder.nbest(trigram_measures.student_t, n_trigrams)

    bigrams = [" ".join(tup) for tup in bigrams]
    trigrams = [" ".join(tup) for tup in trigrams]

    return bigrams + trigrams


def _add_ngrams(intent_records: list[dict]):
    for intent in intent_records:
        text = "\n\n".join(intent["sample_utterances"])
        intent["regexp_partial_match"] = find_ngrams(text, language=args.language)
    return intent_records


def _sample_shots(intent_records: list[dict], n_shots: int, seed: int):
    for intent in intent_records:
        if intent["intent_id"] == -1:
            continue
        intent["sample_utterances"] = random.sample(intent["sample_utterances"], k=n_shots)
    return intent_records


def get_banking77(intent_dataset_train, seed=0, shots_per_intent=5, add_ngrams=False):
    banking77_info = json.load(open("data/banking77_info.json"))
    intent_names = banking77_info["default"]["features"]["label"]["names"]

    all_labels = sorted(intent_dataset_train.unique("label"))
    assert all_labels == list(range(len(intent_names)))

    res = [
        {
            "intent_id": i,
            "intent_name": name,
            "sample_utterances": [],
            "regexp_full_match": [],
            "regexp_partial_match": [],
        }
        for i, name in enumerate(intent_names)
    ]

    for b77_batch in intent_dataset_train.iter(batch_size=16, drop_last_batch=False):
        for txt, intent_id in zip(b77_batch["text"], b77_batch["label"], strict=False):
            res[intent_id]["sample_utterances"].append(txt)

    if add_ngrams:
        res = _add_ngrams(res)

    if shots_per_intent is not None:
        res = _sample_shots(res, shots_per_intent, seed)

    return res


def get_clinc150(intent_dataset_train, seed=0, shots_per_intent=5, add_ngrams=False, oos="ood"):
    intent_names = sorted(intent_dataset_train.unique("labels"))
    oos_intent_id = intent_names.index(oos)
    intent_names = intent_names[:oos_intent_id] + intent_names[oos_intent_id + 1 :] + [intent_names[oos_intent_id]]
    name_to_id = dict(zip(intent_names, range(len(intent_names)), strict=False))
    name_to_id[oos] = -1

    res = [
        {
            "intent_id": i,
            "intent_name": name,
            "sample_utterances": [],
            "regexp_full_match": [],
            "regexp_partial_match": [],
        }
        for name, i in name_to_id.items()
    ]

    for batch in intent_dataset_train.iter(batch_size=16, drop_last_batch=False):
        for txt, name in zip(batch["data"], batch["labels"], strict=False):
            intent_id = name_to_id[name]
            res[intent_id]["sample_utterances"].append(txt)

    if add_ngrams:
        res = _add_ngrams(res)

    if shots_per_intent is not None:
        res = _sample_shots(res, shots_per_intent, seed)

    return res


def get_ru_clinc150(intent_dataset_train, shots_per_intent, oos=42, add_ngrams=False, seed=0):
    all_labels = sorted(intent_dataset_train.unique("intent"))
    assert all_labels == list(range(151))

    in_domain_samples = intent_dataset_train.filter(lambda x: x["intent"] != oos)
    oos_samples = intent_dataset_train.filter(lambda x: x["intent"] == oos)

    res = [
        {
            "intent_id": i,
            "intent_name": None,
            "sample_utterances": [],
            "regexp_full_match": [],
            "regexp_partial_match": [],
        }
        for i in range(150)
    ]

    for batch in in_domain_samples.iter(batch_size=16, drop_last_batch=False):
        for txt, intent_id in zip(batch["text"], batch["intent"], strict=False):
            intent_id -= int(intent_id > oos)
            res[intent_id]["sample_utterances"].append(txt)

    res.append(
        {
            "intent_id": -1,
            "intent_name": "ood",
            "sample_utterances": oos_samples["text"],
            "regexp_full_match": [],
            "regexp_partial_match": [],
        }
    )

    if add_ngrams:
        res = _add_ngrams(res)

    if shots_per_intent is not None:
        res = _sample_shots(res, shots_per_intent, seed)

    return res


def get_snips(intent_dataset_train, label_col, seed=0, shots_per_intent=5, add_ngrams=False):
    intent_names = sorted(intent_dataset_train.unique(label_col))
    name_to_id = dict(zip(intent_names, range(len(intent_names)), strict=False))

    res = [
        {
            "intent_id": i,
            "intent_name": name,
            "sample_utterances": [],
            "regexp_full_match": [],
            "regexp_partial_match": [],
        }
        for i, name in enumerate(intent_names)
    ]

    for batch in intent_dataset_train.iter(batch_size=16, drop_last_batch=False):
        for txt, name in zip(batch["text"], batch[label_col], strict=False):
            intent_id = name_to_id[name]
            res[intent_id]["sample_utterances"].append(txt)

    if add_ngrams:
        res = _add_ngrams(res)

    if shots_per_intent is not None:
        res = _sample_shots(res, shots_per_intent, seed)

    return res


def get_hwu64(hwu_labels, hwu_utterances, seed=0, shots_per_intent=5, add_ngrams=False):
    intent_names = sorted(set(hwu_labels))
    name_to_id = dict(zip(intent_names, range(len(intent_names)), strict=False))

    res = [
        {
            "intent_id": i,
            "intent_name": name,
            "sample_utterances": [],
            "regexp_full_match": [],
            "regexp_partial_match": [],
        }
        for i, name in enumerate(intent_names)
    ]

    for txt, name in zip(hwu_utterances, hwu_labels, strict=False):
        intent_id = name_to_id[name]
        res[intent_id]["sample_utterances"].append(txt)

    if add_ngrams:
        res = _add_ngrams(res)

    if shots_per_intent is not None:
        res = _sample_shots(res, shots_per_intent, seed)

    return res


def get_ru_hwu64(intent_dataset_train, shots_per_intent, add_ngrams=False, seed=0):
    intent_names = sorted(intent_dataset_train.unique("intent"))
    name_to_id = dict(zip(intent_names, range(len(intent_names)), strict=False))

    res = [
        {
            "intent_id": i,
            "intent_name": name,
            "sample_utterances": [],
            "regexp_full_match": [],
            "regexp_partial_match": [],
        }
        for i, name in enumerate(intent_names)
    ]

    for batch in intent_dataset_train.iter(batch_size=16, drop_last_batch=False):
        for txt, name in zip(batch["text"], batch["intent"], strict=False):
            intent_id = name_to_id[name]
            res[intent_id]["sample_utterances"].append(txt)

    if add_ngrams:
        res = _add_ngrams(res)

    if shots_per_intent is not None:
        res = _sample_shots(res, shots_per_intent, seed)

    return res


def get_minds14(intent_dataset_train, shots_per_intent, text_col, add_ngrams=False, seed=0):
    res = [
        {
            "intent_id": i,
            "intent_name": None,
            "sample_utterances": [],
            "regexp_full_match": [],
            "regexp_partial_match": [],
        }
        for i in range(14)
    ]

    for batch in intent_dataset_train.iter(batch_size=16, drop_last_batch=False):
        for txt, intent_id in zip(batch[text_col], batch["intent_class"], strict=False):
            target_list = res[intent_id]["sample_utterances"]
            target_list.append(txt)

    if add_ngrams:
        res = _add_ngrams(res)

    if shots_per_intent is not None:
        res = _sample_shots(res, shots_per_intent, seed)

    return res


def get_massive(intent_dataset_train, shots_per_intent, add_ngrams=False, seed=0):
    intent_names = sorted(intent_dataset_train.unique("label"))
    name_to_id = dict(zip(intent_names, range(len(intent_names)), strict=False))

    res = [
        {
            "intent_id": i,
            "intent_name": name,
            "sample_utterances": [],
            "regexp_full_match": [],
            "regexp_partial_match": [],
        }
        for i, name in enumerate(intent_names)
    ]

    for batch in intent_dataset_train.iter(batch_size=16, drop_last_batch=False):
        for txt, name in zip(batch["text"], batch["label"], strict=False):
            intent_id = name_to_id[name]
            res[intent_id]["sample_utterances"].append(txt)

    if add_ngrams:
        res = _add_ngrams(res)

    if shots_per_intent is not None:
        res = _sample_shots(res, shots_per_intent, seed)

    return res


if __name__ == "__main__":
    import json
    import os
    from argparse import ArgumentParser

    from datasets import load_dataset, load_from_disk

    nltk.download("punkt")
    nltk.download("stopwords")

    parser = ArgumentParser()
    parser.add_argument("--n-shots", type=int, default=5)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--language", choices=["russian", "english"], required=True)
    parser.add_argument("--add-ngrams", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dataset",
        choices=["banking77", "clinc150", "snips", "hwu64", "minds14", "massive"],
    )
    args = parser.parse_args()

    if args.dataset == "banking77":
        if args.language == "russian":
            intent_dataset = load_from_disk("data/RuBanking77")
        elif args.language == "english":
            intent_dataset = load_dataset("PolyAI/banking77")
        else:
            msg = "unsupported language"
            raise ValueError(msg)

        intent_records = get_banking77(
            intent_dataset["train"],
            shots_per_intent=args.n_shots,
            seed=args.seed,
            add_ngrams=args.add_ngrams,
        )
    elif args.dataset == "clinc150":
        if args.language == "russian":
            intent_dataset = load_from_disk("data/RuClinc150")
            func = get_clinc150
        elif args.language == "english":
            intent_dataset = load_dataset("cmaldona/All-Generalization-OOD-CLINC150")
            func = get_ru_clinc150
        else:
            msg = "unsupported language"
            raise ValueError(msg)

        intent_records = func(
            intent_dataset["train"],
            shots_per_intent=args.n_shots,
            seed=args.seed,
            add_ngrams=args.add_ngrams,
        )
    elif args.dataset == "snips":
        if args.language == "russian":
            intent_dataset = load_from_disk("data/RuSnips")
            label_col = "intent"
        elif args.language == "english":
            intent_dataset = load_dataset("benayas/snips")
            label_col = "category"
        else:
            msg = "unsupported language"
            raise ValueError(msg)

        intent_records = get_snips(
            intent_dataset["train"],
            label_col=label_col,
            shots_per_intent=args.n_shots,
            seed=args.seed,
            add_ngrams=args.add_ngrams,
        )
    elif args.dataset == "hwu64":
        if args.language == "russian":
            intent_dataset = load_from_disk("data/RuHWU64")
            intent_records = get_ru_hwu64(
                intent_dataset["train"],
                shots_per_intent=args.n_shots,
                seed=args.seed,
                add_ngrams=args.add_ngrams,
            )
        elif args.language == "english":
            hwu64_labels = open("data/hwu_assets/label.txt").read().split("\n")[:-1]
            hwu64_utterances = open("data/hwu_assets/seq.in").read().split("\n")[:-1]

            intent_records = get_hwu64(
                hwu64_labels,
                hwu64_utterances,
                shots_per_intent=args.n_shots,
                seed=args.seed,
                add_ngrams=args.add_ngrams,
            )
        else:
            msg = "unsupported language"
            raise ValueError(msg)
    elif args.dataset == "minds14":
        intent_dataset = load_dataset("PolyAI/minds14", "ru-RU")
        if args.language == "russian":
            text_col = "transcription"
        elif args.language == "english":
            text_col = "english_transcription"
        else:
            msg = "unsupported language"
            raise ValueError(msg)

        intent_records = get_minds14(
            intent_dataset["train"],
            text_col=text_col,
            shots_per_intent=args.n_shots,
            seed=args.seed,
            add_ngrams=args.add_ngrams,
        )
    elif args.dataset == "massive":
        raise NotImplementedError
        # if args.language == "russian":
        #     intent_dataset = load_dataset("mteb/amazon_massive_intent", 'ru')
        # elif args.language == "english":
        #     intent_dataset = load_dataset("mteb/amazon_massive_intent", 'en')
        # else:
        #     raise ValueError("unsupported language")

        # intent_records = get_massive(
        #     intent_dataset["train"],
        #     shots_per_intent=args.n_shots,
        #     seed=args.seed,
        #     add_ngrams=args.add_ngrams,
        # )

    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json.dump(intent_records, open(args.output_path, "w"), indent=4, ensure_ascii=False)
