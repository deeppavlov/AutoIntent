import nltk
from nltk.collocations import (
    BigramAssocMeasures,
    BigramCollocationFinder,
    TrigramAssocMeasures,
    TrigramCollocationFinder,
)
from nltk.tokenize import word_tokenize



def preprocess(text, language):
    tokens = word_tokenize(text, language=language)
    tokens = ["".join([c.lower() for c in word if c.isalnum()]) for word in tokens]
    tokens = [word for word in tokens if len(word) > 0]
    return tokens


def gen_collocations(text: str, language, n_collocs=3):
    tokens = preprocess(text, language=language)

    bigram_measures = BigramAssocMeasures()
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(bigram_measures.student_t, n_collocs)

    trigram_measures = TrigramAssocMeasures()
    trigram_finder = TrigramCollocationFinder.from_words(tokens)
    trigrams = trigram_finder.nbest(trigram_measures.student_t, n_collocs)

    bigrams = [' '.join(tup) for tup in bigrams]
    trigrams = [' '.join(tup) for tup in trigrams]

    return bigrams, trigrams


if __name__ == "__main__":
    import json
    import os
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, default="experiments/regexp/data/banking77.json")
    parser.add_argument("--n-collocs", type=int, default=3)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    args = parser.parse_args()

    nltk.download('punkt_tab')

    intent_records = json.load(open(args.input_path))

    for intent in intent_records:
        text = "\n\n".join(intent["sample_utterances"])
        bigrams, trigrams = gen_collocations(text, language=args.language, n_collocs=args.n_collocs)
        intent["regexp_partial_match"] = bigrams + trigrams
    
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json.dump(intent_records, open(args.output_path, "w"), indent=4, ensure_ascii=False)
