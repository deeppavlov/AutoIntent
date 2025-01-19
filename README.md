# AutoIntent

<img align="left" width="100" height="100" src="logo/square-light.svg#gh-light-mode-only">
<img align="left" width="100" height="100" src="logo/square-dark.svg#gh-dark-mode-only">

Auto ML for intent classification.

Documentation: [deeppavlov.github.io/AutoIntent](https://deeppavlov.github.io/AutoIntent/).

> The project is under active development.

## Installation

```bash
pip install autointent
```

## About

**AutoIntent** is an open source tool for automatic configuration of a text classification pipeline for intent prediction.

The task of intent detection is one of the main subtasks in creating task-oriented dialogue systems, along with scriptwriting and slot filling. AutoIntent project offers users the following:

- A convenient library of methods for intent classification that can be used in a sklearn-like "fit-predict" format.
- An AutoML approach to creating classifiers, where the only thing needed is to upload a set of labeled data.

Example of building an intent classifier in a couple of lines of code:

```python
from autointent import Pipeline, Dataset

dataset = Dataset.from_json(path_to_json)
pipeline = Pipeline.default_optimizer(multilabel=False)
pipeline.fit(dataset)
pipeline.predict(["show me my latest recent transactions"])
```
