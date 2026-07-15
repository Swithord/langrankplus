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

SIB_LABELS = (
    "science/technology",
    "travel",
    "politics",
    "sports",
    "health",
    "entertainment",
    "geography",
)

LABEL_TO_ID = {
    label: index
    for index, label in enumerate(SIB_LABELS)
}


def preprocess_dataset(
    dataset,
    tokenizer,
    args,
):
    max_length = args.max_length

    def tokenize(batch):
        unknown = sorted(
            set(batch["category"])
            - set(LABEL_TO_ID)
        )

        if unknown:
            raise ValueError(
                f"Unknown SIB-200 categories: {unknown}"
            )

        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

        encoded["labels"] = [
            LABEL_TO_ID[label]
            for label in batch["category"]
        ]

        return encoded

    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        **map_kwargs(
            args,
            "Tokenising SIB-200",
        ),
    )


TASK = LlamaTask(
    name="sib200",
    description=(
        "Fine-tune Llama 3.1 8B with QLoRA on one "
        "SIB-200 transfer language and evaluate it "
        "on every local task language."
    ),
    model_class=AutoModelForSequenceClassification,
    peft_task_type=TaskType.SEQ_CLS,
    labels=SIB_LABELS,
    preprocess_dataset=preprocess_dataset,
    data_collator_factory=(
        lambda tokenizer: DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        )
    ),
    compute_metrics=macro_f1_metrics,
    epochs=30,
    learning_rate=2e-4,
    train_batch_size=4,
    eval_batch_size=8,
    gradient_accumulation_steps=4,
    eval_steps=20,
    patience=10,
)


if __name__ == "__main__":
    run_task(TASK)