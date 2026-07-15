from __future__ import annotations

from peft import TaskType
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

from llama_utils import (
    LlamaTask,
    macro_f1_metrics,
    map_kwargs,
    run_task,
)

XNLI_LABELS = (
    "entailment",
    "neutral",
    "contradiction",
)


def preprocess_dataset(
    dataset,
    tokenizer,
    args,
):
    max_length = args.max_length

    def tokenize(batch):
        labels = [
            int(label)
            for label in batch["label"]
        ]

        if any(
            label < 0 or label >= len(XNLI_LABELS)
            for label in labels
        ):
            raise ValueError(
                "XNLI contains a label outside {0, 1, 2}."
            )

        encoded = tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=max_length,
        )

        encoded["labels"] = labels

        return encoded

    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        **map_kwargs(
            args,
            "Tokenising XNLI",
        ),
    )


TASK = LlamaTask(
    name="xnli",
    description=(
        "Fine-tune Llama 3.1 8B with QLoRA on one "
        "XNLI transfer language and evaluate it "
        "on every local task language."
    ),
    model_class=AutoModelForSequenceClassification,
    peft_task_type=TaskType.SEQ_CLS,
    labels=XNLI_LABELS,
    preprocess_dataset=preprocess_dataset,
    data_collator_factory=(
        lambda tokenizer: DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        )
    ),
    compute_metrics=macro_f1_metrics,
    epochs=3,
    learning_rate=1e-4,
    train_batch_size=4,
    eval_batch_size=8,
    gradient_accumulation_steps=8,
    eval_steps=2000,
    patience=5,
)


if __name__ == "__main__":
    run_task(TASK)