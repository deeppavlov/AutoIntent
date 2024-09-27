"""
This examples trains a CrossEncoder for the STSbenchmark task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continuous labels 0...1 to indicate the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_stsbenchmark.py
"""

import sys

sys.path.append("/home/voorhs/repos/AutoIntent")

import itertools as it
from random import shuffle

import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder, InputExample
from torch import nn
from transformers import AutoModelForSequenceClassification


def construct_samples(texts, labels, balancing_factor: int = None) -> list[InputExample]:
    samples = [[], []]

    for (i, text1), (j, text2) in it.combinations(enumerate(texts), 2):
        pair = [text1, text2]
        label = int(labels[i] == labels[j])
        sample = InputExample(texts=pair, label=label)
        samples[label].append(sample)
    shuffle(samples[0])
    shuffle(samples[1])

    if balancing_factor is not None:
        i_min = min([0, 1], key=lambda i: len(samples[i]))
        i_max = 1 - i_min
        min_length = len(samples[i_min])
        res = samples[i_min][:min_length] + samples[i_max][: min_length * balancing_factor]
    else:
        res = samples[0] + samples[1]

    return res


def freeze_encoder(model: AutoModelForSequenceClassification):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad_(False)

    # Unfreeze the classifier layer
    for param in model.classifier.parameters():
        param.requires_grad_(True)


class LogLoss(nn.Module):
    def __init__(self, model: CrossEncoder, label_smoothing=0.0):
        super().__init__()

        self.model = model
        self.label_smoothing = label_smoothing

    def forward(self, sentence_features: torch.Tensor, labels: torch.Tensor):
        """
        Arguments
        ---
        - `sentence_features`: torch.Tensor of shape (batch_size,), predicted probabilities for binary classification
        - `labels`: torch.Tensor of shape (batch_size,), true binary labels
        """
        labels = labels.float()
        smoothed_targets = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        loss = F.binary_cross_entropy_with_logits(sentence_features, smoothed_targets)
        return loss


if __name__ == "__main__":
    import json
    import logging
    import math
    from datetime import datetime

    from sentence_transformers import LoggingHandler
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import (
        CEBinaryClassificationEvaluator,
    )
    from torch.utils.data import DataLoader

    from autointent.data_handler import split_sample_utterances

    #### Just some code to print debug information to stdout
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    logger = logging.getLogger(__name__)
    #### /print debug information to stdout

    # Define our Cross-Encoder
    train_batch_size = 16
    num_epochs = 4
    model_save_path = "experiments/cross-encoder-training/logs/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    model = CrossEncoder("llmrails/ember-v1", num_labels=1)
    freeze_encoder(model.model)

    # Read dataset
    logger.info("Read banking77 train dataset")
    dataset_path = "data/intent_records/banking77.json"
    intent_records = json.load(open(dataset_path))
    (n_classes, utterances_train, utterances_test, labels_train, labels_test) = split_sample_utterances(intent_records)

    train_samples = construct_samples(utterances_train, labels_train, balancing_factor=1)
    test_samples = construct_samples(utterances_test, labels_test, balancing_factor=1)

    # We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
    train_dataloader = DataLoader(
        train_samples,
        shuffle=True,
        batch_size=train_batch_size,
        drop_last=True,
        num_workers=0,  # don't change it!
    )

    # We add an evaluator, which evaluates the performance during training
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_samples, name="test")
    evaluator(model)

    # Configure the training (10% of train data for warm-up)
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    logger.info(f"Warmup-steps: {warmup_steps}")

    # Train the model
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        optimizer_params={"lr": 2e-5},
        loss_fct=LogLoss(model, label_smoothing=0.2),
    )

    ##### Load model and eval on test set
    model = CrossEncoder(model_save_path)

    evaluator(model)
