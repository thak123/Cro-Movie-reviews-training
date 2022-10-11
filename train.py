from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer,
)
from datasets import (
    Dataset,
    load_dataset,
    load_metric,
    concatenate_datasets,
    DatasetDict,
)
import torch
import torch.nn as nn
import numpy as np
import os
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
import pandas as pd

import torch

torch.manual_seed(0)
import random

random.seed(0)
import numpy as np

np.random.seed(0)

os.environ["WANDB_DISABLED"] = "true"


# model_name = "xlm-roberta-base"
# model_name ="classla/bcms-bertic" 
model_name ="EMBEDDIA/crosloengual-bert" #
tokenizer = AutoTokenizer.from_pretrained(model_name)

config = AutoConfig.from_pretrained(
    model_name, num_labels=3, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2
)
model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=model_name, config=config,
)


def compute_metrics(eval_pred):
    f1 = load_metric("f1")
    accuracy = load_metric("accuracy")
    precision = load_metric("precision")
    recall = load_metric("recall")
    metrics_dict = {}
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    metrics_dict.update(
        f1.compute(predictions=predictions, references=labels, average="macro")
    )
    metrics_dict.update(accuracy.compute(predictions=predictions, references=labels))
    metrics_dict.update(
        precision.compute(predictions=predictions, references=labels, average="macro")
    )
    metrics_dict.update(
        recall.compute(predictions=predictions, references=labels, average="macro")
    )
    return metrics_dict


def tokenize_function(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=512
    )


def return_text(path):
    with open(path) as inputfile:
        lines = []
        for line in inputfile:
            line = line.strip()
            if line:
                lines.append(line)
    return lines


if __name__ == "__main__":

    dataset_path = "/home/gaurishthakkar/projects/Cro-Movie-reviews-training/data/"

    train_df_hr = pd.read_csv(dataset_path + "train.tsv", sep="\t")
    train_df_hr = train_df_hr[train_df_hr["label"] != 3]

    train_df_cz_pos = pd.DataFrame({"text": return_text(dataset_path + "positive.txt")})
    train_df_cz_pos["label"] = 2
    # train_df_cz_pos = train_df_cz_pos.head(2000)

    train_df_cz_neu = pd.DataFrame({"text": return_text(dataset_path + "neutral.txt")})
    train_df_cz_neu["label"] = 1
    # train_df_cz_neu = train_df_cz_neu.head(2000)

    train_df_cz_neg = pd.DataFrame({"text": return_text(dataset_path + "negative.txt")})
    train_df_cz_neg["label"] = 0
    # train_df_cz_neg = train_df_cz_neg.head(2000)

    # train_df = pd.concat(
    #     [train_df_hr, train_df_cz_pos, train_df_cz_neu, train_df_cz_neg]
    # )
    train_df = train_df_hr
    print("label_values:", train_df["label"].value_counts())

    test_df = pd.read_csv(dataset_path + "eval.tsv", sep="\t")
    test_df = test_df[test_df["label"] != 3]

    y = np.array(train_df["label"])

    print("Unique-labels", np.unique(y))

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y), y=y
    )

    # oversample the under represented classes
    # train_df = oversample(train_df, "label")

    ds = DatasetDict(
        {"train": Dataset.from_pandas(train_df), "test": Dataset.from_pandas(test_df)}
    )
    tokenized_datasets = ds.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    from datetime import datetime

    training_args = TrainingArguments(
        output_dir=f"./logs/" + str(datetime.now()),
        eval_steps=500,
        overwrite_output_dir=True,
        per_device_train_batch_size=16,
        do_train=True,
        do_eval=True,
        warmup_steps=0,  # 500,
        learning_rate=1e-05,  # 1e-5,
        weight_decay=0.02,
        num_train_epochs=10,
        save_total_limit=1,
        save_strategy="steps",
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        report_to="tensorboard",
        seed=10,
    )

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (suppose one has 3 labels with different weights)
            loss_fct = nn.CrossEntropyLoss(
                # weight=torch.tensor(class_weights, dtype=torch.float32,device="cuda")
            )
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels), labels.view(-1)
            )
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=4,
            )
        ],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(
        f"./best_model"
    )  # TODO HARDCODED save folder should follow the method and the language naming

    print(trainer.evaluate())
