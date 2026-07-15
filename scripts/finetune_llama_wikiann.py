from __future__ import annotations

import numpy as np
from peft import TaskType
from seqeval.metrics import classification_report
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)

from llama_utils import (
    LlamaTask,
    map_kwargs,
    run_task,
)

WIKIANN_LABELS = (
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
)

ID_TO_LABEL = {
    index: label
    for index, label in enumerate(WIKIANN_LABELS)
}

ENTITY_TYPES = (
    "PER",
    "ORG",
    "LOC",
)


def preprocess_dataset(
    dataset,
    tokenizer,
    args,
):
    if not tokenizer.is_fast:
        raise RuntimeError(
            "WikiAnn alignment requires a fast tokenizer."
        )

    max_length = args.max_length

    def tokenize_and_align(batch):
        encoded = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )

        aligned_labels = []

        for batch_index, word_labels in enumerate(
            batch["ner_tags"]
        ):
            word_ids = encoded.word_ids(
                batch_index=batch_index
            )

            previous_word_id = None
            labels = []

            for word_id in word_ids:
                if (
                    word_id is None
                    or word_id == previous_word_id
                ):
                    labels.append(-100)
                else:
                    label = int(
                        word_labels[word_id]
                    )

                    if (
                        label < 0
                        or label >= len(WIKIANN_LABELS)
                    ):
                        raise ValueError(
                            f"Invalid WikiAnn label ID: {label}"
                        )

                    labels.append(label)

                previous_word_id = word_id

            aligned_labels.append(labels)

        encoded["labels"] = aligned_labels

        return encoded

    return dataset.map(
        tokenize_and_align,
        remove_columns=dataset.column_names,
        **map_kwargs(
            args,
            "Tokenising and aligning WikiAnn",
        ),
    )


def compute_metrics(
    eval_prediction,
):
    logits, labels = eval_prediction

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = (
        np.asarray(logits)
        .argmax(axis=-1)
    )

    true_predictions = []
    true_labels = []

    for prediction_row, label_row in zip(
        predictions,
        labels,
    ):
        kept_predictions = []
        kept_labels = []

        for prediction, label in zip(
            prediction_row,
            label_row,
        ):
            if label != -100:
                kept_predictions.append(
                    ID_TO_LABEL[int(prediction)]
                )
                kept_labels.append(
                    ID_TO_LABEL[int(label)]
                )

        true_predictions.append(
            kept_predictions
        )
        true_labels.append(
            kept_labels
        )

    report = classification_report(
        true_labels,
        true_predictions,
        output_dict=True,
        zero_division=0,
    )

    type_f1_scores = [
        float(
            report[entity_type]["f1-score"]
        )
        for entity_type in ENTITY_TYPES
        if entity_type in report
    ]

    if type_f1_scores:
        macro_f1 = float(
            np.mean(type_f1_scores)
        )
    else:
        macro_f1 = 0.0

    return {"f1": macro_f1}


TASK = LlamaTask(
    name="wikiann",
    description=(
        "Fine-tune Llama 3.1 8B with QLoRA on one "
        "WikiAnn transfer language and evaluate it "
        "on every local task language."
    ),
    model_class=AutoModelForTokenClassification,
    peft_task_type=TaskType.TOKEN_CLS,
    labels=WIKIANN_LABELS,
    preprocess_dataset=preprocess_dataset,
    data_collator_factory=(
        lambda tokenizer: DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        )
    ),
    compute_metrics=compute_metrics,
    epochs=30,
    learning_rate=2e-4,
    train_batch_size=4,
    eval_batch_size=8,
    gradient_accumulation_steps=8,
    eval_steps=200,
    patience=5,
)


if __name__ == "__main__":
    run_task(TASK)