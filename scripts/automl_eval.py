from datasets import load_dataset, concatenate_datasets
import pandas as pd
import logging
import numpy as np
import argparse
from sklearn.metrics import classification_report

import wandb

logging.basicConfig(level="INFO")


def load_data(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a dataset from the Hugging Face datasets library.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        DatasetDict: A dictionary containing the train, validation, and test splits of the dataset.
    """
    # Load the dataset
    dataset = load_dataset(dataset_name)

    if "train_0" in dataset:
        for col in ["train", "validation"]:
            dataset[col] = concatenate_datasets([dataset[f"{col}_0"], dataset[f"{col}_1"]])
            dataset.pop(f"{col}_0")
            dataset.pop(f"{col}_1")

    train_data = dataset["train"]
    test_data = dataset["test"]

    train_df = train_data.to_pandas()
    max_label = train_df["label"].max()
    train_df.loc[train_df["label"].isna(), "label"] = max_label + 1

    test_df = test_data.to_pandas()
    test_df.loc[test_df["label"].isna(), "label"] = max_label + 1
    return train_df, test_df


def evalute_fedot(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Train a Fedot model on the provided training and testing data.

    Args:
        train_df (pd.DataFrame): The training data.
        test_df (pd.DataFrame): The testing data.
    """
    # !pip install fedot
    from fedot.api.main import Fedot

    X_train, y_train = train_df[["utterance"]], train_df["label"].astype(int)
    X_test, y_test = test_df[["utterance"]], test_df["label"].astype(int)
    model = Fedot(problem="classification", timeout=5, preset="best_quality", n_jobs=-1)
    model.fit(features=X_train, target=y_train)
    prediction = model.predict(features=X_test)
    return prediction


def evaluate_h2o(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train an H2O model on the provided training and testing data.

    Args:
        train_df (pd.DataFrame): The training data.
        test_df (pd.DataFrame): The testing data.
    """
    # !pip install h2o
    import h2o
    from h2o.estimators import H2OGradientBoostingEstimator
    from h2o.estimators.word2vec import H2OWord2vecEstimator
    from h2o.automl import H2OAutoML

    max_models: int = 20
    max_runtime_secs: int = 600
    seed: int = 42

    h2o.init()

    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)
    train_h2o["label"] = train_h2o["label"].asfactor()
    test_h2o["label"] = test_h2o["label"].asfactor()
    train, valid = train_h2o.split_frame(ratios=[0.8])
    text_col = "utterance"
    label_col = "label"
    train_tokens = train[text_col].tokenize("\\s+")
    valid_tokens = valid[text_col].tokenize("\\s+")
    test_tokens = test_h2o[text_col].tokenize(
        "\\s+"
    )  # Word2Vec needs token lists :contentReference[oaicite:0]{index=0}

    w2v_model = H2OWord2vecEstimator(sent_sample_rate=0.0, epochs=10)
    w2v_model.train(training_frame=train_tokens)

    train_vecs = w2v_model.transform(train_tokens, aggregate_method="AVERAGE")
    valid_vecs = w2v_model.transform(valid_tokens, aggregate_method="AVERAGE")
    test_vecs = w2v_model.transform(test_tokens, aggregate_method="AVERAGE")

    train_ext = train_vecs.cbind(train[label_col])
    valid_ext = valid_vecs.cbind(valid[label_col])
    test_ext = test_vecs.cbind(test_h2o[label_col])

    x_cols = train_vecs.columns
    y_col = label_col

    # 9. Run H2OAutoML
    aml = H2OAutoML(
        max_models=max_models,
        max_runtime_secs=max_runtime_secs,
        seed=seed,
        balance_classes=True,
        sort_metric="mean_per_class_error",
    )
    aml.train(x=x_cols, y=y_col, training_frame=train_ext, validation_frame=valid_ext, leaderboard_frame=test_ext)

    preds = aml.leader.predict(test_ext)
    return preds["predict"]


def evaluate_lama(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a LAMA model on the provided training and testing data.

    Args:
        train_df (pd.DataFrame): The training data.
        test_df (pd.DataFrame): The testing data.
    """
    # !pip install lightautoml[nlp]
    from lightautoml.automl.presets.text_presets import TabularNLPAutoML
    from lightautoml.tasks import Task
    # pytorch<2.7.0
    # https://github.com/sb-ai-lab/LightAutoML/issues/173

    automl = TabularNLPAutoML(task=Task(name="multiclass", metric="f1_macro"))
    automl.fit_predict(train_df, roles={"target": "label"})
    test_preds = automl.predict(test_df).data
    return np.argmax(test_preds, axis=-1)


def evaluate_gama(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a GAMA model on the provided training and testing data.

    Args:
        train_df (pd.DataFrame): The training data.
        test_df (pd.DataFrame): The testing data.
    """
    # NOT WORKING
    # ValueError: population must be at least size 3 for a pair to be selected
    raise NotImplementedError("GAMA is not working yet.")
    # !pip install gama
    from gama import GamaClassifier

    automl = GamaClassifier(max_total_time=180, store="nothing")
    automl.fit(train_df[["utterance"]], train_df[["label"]])


def evaluate_glueon(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a GlueOn model on the provided training and testing data.

    Args:
        train_df (pd.DataFrame): The training data.
        test_df (pd.DataFrame): The testing data.
    """
    #!pip install autogluon
    from autogluon.multimodal import MultiModalPredictor
    import uuid

    model_path = f"/tmp/{uuid.uuid4().hex}-automm_sst"
    predictor = MultiModalPredictor(label="label", problem_type="multiclass", eval_metric="acc", path=model_path)
    predictor.fit(train_df, time_limit=180)
    predictions = predictor.predict(test_df)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate AutoML models on a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        help="The name of the dataset to evaluate.",
    )
    parser.add_argument(
        "--framework",
        type=str,
        choices=["fedot", "h2o", "lama", "gama", "glueon"],
        help="The name of the model to evaluate.",
    )
    args = parser.parse_args()
    dataset_name = args.dataset
    framework = args.framework
    run = wandb.init(
        project="AutoML-Eval",
        name=f"eval-{dataset_name}-{framework}",
        tags=[dataset_name, framework],
        config={
            "dataset": dataset_name,
            "framework": framework,
        },
    )
    # Load the dataset
    train_df, test_df = load_data(dataset_name)

    # Evaluate the model
    if framework == "fedot":
        predictions = evalute_fedot(train_df, test_df)
    elif framework == "h2o":
        predictions = evaluate_h2o(train_df, test_df)
    elif framework == "lama":
        predictions = evaluate_lama(train_df, test_df)
    elif framework == "gama":
        predictions = evaluate_gama(train_df, test_df)
    elif framework == "glueon":
        predictions = evaluate_glueon(train_df, test_df)
    else:
        raise ValueError(f"Unknown framework: {framework}")
    # Log the predictions
    run.log({"predictions": wandb.Table(dataframe=pd.DataFrame(predictions))})
    # Log the classification report
    report = classification_report(test_df["label"], predictions, output_dict=True)
    run.log(report)
    # Finish the run
    run.finish()


if __name__ == "__main__":
    main()
