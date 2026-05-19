# AutoIntent

<img align="left" width="100" height="100" src="logo/square-light.svg#gh-light-mode-only">
<img align="left" width="100" height="100" src="logo/square-dark.svg#gh-dark-mode-only">

Auto ML for intent classification.

Documentation: [deeppavlov.github.io/AutoIntent](https://deeppavlov.github.io/AutoIntent/).

Changelog: [CHANGELOG.md](./CHANGELOG.md).

The project is under active development.

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
pipeline = Pipeline.from_preset("classic-light")
pipeline.fit(dataset)
pipeline.predict(["show me my latest transactions"])
```

## Cite

If you find our work useful, please cite our EMNLP 2025 [paper](https://arxiv.org/abs/2509.21138):
```
@misc{alekseev2025autointentautomltextclassification,
      title={AutoIntent: AutoML for Text Classification}, 
      author={Ilya Alekseev and Roman Solomatin and Darina Rustamova and Denis Kuznetsov},
      year={2025},
      eprint={2509.21138},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.21138}, 
}
```

## Disclaimer

This project is in development phase. Bugs and breaking changes are expected. Contributions and feedback are welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md).

## Credits

Logo designed by [nkognit0](https://github.com/nkognit0).
