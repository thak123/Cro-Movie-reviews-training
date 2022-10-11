import numpy as np
import os

os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    logging,
)
from datasets import set_caching_enabled

logging.set_verbosity(logging.ERROR)

set_caching_enabled(False)



model_name = "EMBEDDIA/crosloengual-bert"

# model_name = "classla/bcms-bertic" #"EMBEDDIA/crosloengual-bert" #"xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

config = AutoConfig.from_pretrained(model_name)


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




dataset_path = (
    "/home/gaurishthakkar/projects/Cro-Movie-reviews-training/data/test.tsv"
)

ds = load_dataset("csv", delimiter="\t", data_files={"test":dataset_path })


print(":", len(ds["test"]))

tokenized_datasets = ds.map(tokenize_function, batched=True, load_from_cache_file=False)
tokenized_datasets = tokenized_datasets.filter(lambda example: example["label"] !=3)

eval_dataset = tokenized_datasets["test"]

model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path="/home/gaurishthakkar/projects/Cro-Movie-reviews-training/best_model_92",
    num_labels=3,
)


test_args = TrainingArguments(
    output_dir="./logs",
    do_train=False,
    do_eval=True,
    logging_dir="logs",
    report_to="tensorboard",
    disable_tqdm=True,
)
trainer = Trainer(
    model=model,
    args=test_args,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
from transformers.trainer_callback import PrinterCallback

trainer.remove_callback(PrinterCallback)

print(trainer.evaluate())
predictions = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predictions.predictions, axis=-1)


with open("logs/predictions.tsv", "w") as outputfile:
    outputfile.write("\t".join(["text","label","pred"]) + "\n")

    for i in zip(
        tokenized_datasets["test"]["text"], tokenized_datasets["test"]["label"], preds
    ):
        outputfile.write("\t".join([i[0], str(i[1]), str(i[2])]) + "\n")
