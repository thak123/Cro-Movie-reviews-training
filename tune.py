import os
from typing import Callable, Dict
import numpy as np
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.examples.pbt_transformers.utils import (
    build_compute_metrics_fn,
    download_data,
)

import pandas as pd
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest.hyperopt import HyperOptSearch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    EvalPrediction,
    TrainingArguments,
)
from datasets import (
    Dataset,
    load_dataset,
    load_metric,
    concatenate_datasets,
    DatasetDict,
)
os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def tune_transformer(num_samples=8, gpus_per_trial=0, smoke_test=False):
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    f1 = load_metric("f1")
    accuracy = load_metric("accuracy")
    precision = load_metric("precision")
    recall = load_metric("recall")

    def compute_metrics(eval_pred):
        metrics_dict = {}
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        metrics_dict.update(
            f1.compute(predictions=predictions, references=labels, average="macro")
        )
        metrics_dict.update(
            accuracy.compute(predictions=predictions, references=labels)
        )
        metrics_dict.update(
            precision.compute(
                predictions=predictions, references=labels, average="macro"
            )
        )
        metrics_dict.update(
            recall.compute(predictions=predictions, references=labels, average="macro")
        )
        return metrics_dict

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )

    dataset_path = (
        "/home/gaurishthakkar/projects/Cro-Movie-reviews-training/data/"
    )
    # Read the datase files:

    train_df = pd.read_csv(dataset_path + "train.tsv", sep="\t")
    eval_df = pd.read_csv(dataset_path + "eval.tsv", sep="\t")

    # test_df = pd.read_csv(dataset_path + "test.tsv", sep="\t")

    ds = DatasetDict(
        {"train": Dataset.from_pandas(train_df), "test": Dataset.from_pandas(eval_df)}
    )
    tokenized_datasets = ds.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"].shard(index=1, num_shards=10)
    eval_dataset = tokenized_datasets["test"]

    def get_model():
        return AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=4
        )

    training_args = TrainingArguments(
        output_dir="./logs",
        learning_rate=1e-5,  # config
        do_train=True,
        do_eval=True,
        no_cuda=gpus_per_trial <= 0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=2,  # config
        max_steps=-1,
        per_device_train_batch_size=16,  # config
        per_device_eval_batch_size=16,  # config
        warmup_steps=0,
        weight_decay=0.1,  # config
        logging_dir="./logs",
        skip_memory_metrics=True,
        report_to="none",
    )

    trainer = Trainer(
        model_init=get_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    tune_config = {
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "num_train_epochs": tune.choice([2, 3, 4, 5]),
        "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_f1",
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations={
            "weight_decay": tune.uniform(0.0, 0.3),
            "learning_rate": tune.uniform(1e-5, 5e-5),
            "per_device_train_batch_size": [4, 8, 16],
        },
    )

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=[ "eval_f1", "eval_accuracy", "eval_loss", "epoch", "training_iteration"],
    )
    trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        n_trials=num_samples,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        scheduler=scheduler,
        keep_checkpoints_num=1,
        checkpoint_score_attr="training_iteration",
        stop={"training_iteration": 1} if smoke_test else None,
        progress_reporter=reporter,
        local_dir="./ray_results/",
        name="tune_transformer_pbt",
        log_to_file=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Address to use for Ray. "
        'Use "auto" for cluster. '
        "Defaults to None for local.",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using " "Ray Client.",
    )

    args, _ = parser.parse_known_args()

    if args.smoke_test:
        ray.init()
    elif args.server_address:
        ray.init(f"ray://{args.server_address}")
    else:
        ray.init(args.ray_address)

    if args.smoke_test:
        tune_transformer(num_samples=1, gpus_per_trial=1, smoke_test=True)
    else:
        # You can change the number of GPUs here:
        tune_transformer(num_samples=8, gpus_per_trial=1)