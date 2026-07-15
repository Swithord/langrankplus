from __future__ import annotations

import argparse
import fcntl
import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from sklearn.metrics import f1_score
from torch import nn
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
RESULT_COLUMNS = ["task_lang", "transfer_lang", "f1_score"]

LLAMA_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass(frozen=True)
class LlamaTask:
    name: str
    description: str
    model_class: type
    peft_task_type: TaskType
    labels: tuple[str, ...]
    preprocess_dataset: Callable[[Any, Any, argparse.Namespace], Any]
    data_collator_factory: Callable[[Any], Any]
    compute_metrics: Callable[[Any], dict[str, float]]
    epochs: float
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    eval_steps: int
    patience: int
    max_length: int = 256


def build_parser(task: LlamaTask) -> argparse.ArgumentParser:
    default_num_proc = min(
        8,
        int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
    )

    parser = argparse.ArgumentParser(description=task.description)

    parser.add_argument(
        "--lang",
        required=True,
        help="Transfer language to fine-tune on.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load the saved adapter and skip fine-tuning.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Trainer checkpoint directory from which to resume.",
    )
    parser.add_argument(
        "--task-langs",
        nargs="+",
        default=None,
        help=(
            "Optional subset of task languages to evaluate. "
            "The default is all local languages."
        ),
    )

    parser.add_argument(
        "--base-model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("models"),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs"),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("data"),
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=task.epochs,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=task.learning_rate,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=task.train_batch_size,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=task.eval_batch_size,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=task.gradient_accumulation_steps,
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=task.eval_steps,
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=task.patience,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=task.max_length,
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16"),
        default="auto",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=default_num_proc,
    )
    parser.add_argument(
        "--eval-accumulation-steps",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
    )

    return parser


def macro_f1_metrics(
    eval_prediction: Any,
) -> dict[str, float]:
    logits, labels = eval_prediction

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.asarray(logits).argmax(axis=-1)

    score = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {"f1": float(score)}


def map_kwargs(
    args: argparse.Namespace,
    description: str,
) -> dict[str, Any]:
    return {
        "batched": True,
        "num_proc": max(1, args.num_proc),
        "load_from_cache_file": not args.overwrite_cache,
        "desc": description,
    }


def limit_dataset(
    dataset: Any,
    maximum: int | None,
) -> Any:
    if maximum is None or maximum >= len(dataset):
        return dataset

    if maximum <= 0:
        raise ValueError("Sample limits must be positive.")

    return dataset.select(range(maximum))


def _hub_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _resolve_dtype(
    precision: str,
) -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA requires a CUDA GPU for this implementation."
        )

    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "The selected GPU does not support bfloat16."
            )
        return torch.bfloat16

    if precision == "fp16":
        return torch.float16

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def _load_tokenizer(
    source: str | Path,
) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        token=_hub_token(),
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                "The tokenizer has neither a padding token nor an EOS token."
            )

        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def _replace_task_head(
    model: Any,
    num_labels: int,
) -> str:
    head_name = next(
        (
            name
            for name in ("score", "classifier")
            if hasattr(model, name)
        ),
        None,
    )

    if head_name is None:
        raise RuntimeError(
            "Could not find the model's classification head."
        )

    old_head = getattr(model, head_name)

    if not hasattr(old_head, "in_features"):
        raise RuntimeError(
            f"The {head_name!r} head is not a linear layer."
        )

    new_head = nn.Linear(
        old_head.in_features,
        num_labels,
        bias=getattr(old_head, "bias", None) is not None,
    )

    if hasattr(model, "_init_weights"):
        model._init_weights(new_head)

    head_device = old_head.weight.device

    if head_device.type == "meta":
        head_device = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )

    new_head = new_head.to(
        device=head_device,
        dtype=torch.float32,
    )

    setattr(model, head_name, new_head)

    return head_name


def _load_quantized_base(
    task: LlamaTask,
    args: argparse.Namespace,
    tokenizer: Any,
    compute_dtype: torch.dtype,
) -> tuple[Any, str]:
    id2label = {
        index: label
        for index, label in enumerate(task.labels)
    }
    label2id = {
        label: index
        for index, label in id2label.items()
    }

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = task.model_class.from_pretrained(
        args.base_model,
        num_labels=len(task.labels),
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map={"": torch.cuda.current_device()},
        low_cpu_mem_usage=True,
        token=_hub_token(),
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    head_name = _replace_task_head(
        model,
        len(task.labels),
    )

    return model, head_name


def _build_training_model(
    task: LlamaTask,
    args: argparse.Namespace,
    tokenizer: Any,
    compute_dtype: torch.dtype,
) -> Any:
    model, head_name = _load_quantized_base(
        task,
        args,
        tokenizer,
        compute_dtype,
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    peft_config = LoraConfig(
        task_type=task.peft_task_type,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LLAMA_LORA_TARGETS,
        modules_to_save=[head_name],
        bias="none",
    )

    model = get_peft_model(
        model,
        peft_config,
    )

    model.print_trainable_parameters()

    return model


def _load_adapter_model(
    task: LlamaTask,
    args: argparse.Namespace,
    tokenizer: Any,
    compute_dtype: torch.dtype,
    adapter_dir: Path,
) -> Any:
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"No saved adapter found at {adapter_dir}."
        )

    base_model, _ = _load_quantized_base(
        task,
        args,
        tokenizer,
        compute_dtype,
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable=False,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()

    return model


def _training_arguments(
    args: argparse.Namespace,
    compute_dtype: torch.dtype,
    checkpoint_dir: Path,
    logging_dir: Path,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(checkpoint_dir),
        logging_dir=str(logging_dir),

        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",

        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=args.logging_steps,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,

        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),

        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        optim="paged_adamw_8bit",
        max_grad_norm=args.max_grad_norm,

        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        tf32=True,

        gradient_checkpointing=True,
        eval_accumulation_steps=(
            args.eval_accumulation_steps
        ),

        dataloader_num_workers=min(
            4,
            max(1, args.num_proc),
        ),
        group_by_length=True,

        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        label_names=["labels"],
        remove_unused_columns=True,
    )


def _prediction_arguments(
    args: argparse.Namespace,
    compute_dtype: torch.dtype,
    output_dir: Path,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),

        per_device_eval_batch_size=args.eval_batch_size,

        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        tf32=True,

        eval_accumulation_steps=(
            args.eval_accumulation_steps
        ),

        dataloader_num_workers=min(
            4,
            max(1, args.num_proc),
        ),

        report_to="none",
        seed=args.seed,
        label_names=["labels"],
        remove_unused_columns=True,
    )


def _dataset_languages(
    data_dir: Path,
) -> list[str]:
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_dir}"
        )

    languages = sorted(
        path.name
        for path in data_dir.iterdir()
        if (
            path.is_dir()
            and (path / "dataset_dict.json").is_file()
        )
    )

    if not languages:
        raise RuntimeError(
            f"No saved DatasetDict language folders found in {data_dir}."
        )

    return languages


def _validate_languages(
    requested: Sequence[str] | None,
    available: Sequence[str],
) -> list[str]:
    if requested is None:
        return list(available)

    missing = sorted(
        set(requested) - set(available)
    )

    if missing:
        raise ValueError(
            "Task-language folders not found: "
            + ", ".join(missing)
        )

    return list(dict.fromkeys(requested))


def _write_results(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock_dir = output_path.parent / ".locks"
    lock_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock_path = (
        lock_dir
        / f"{output_path.name}.lock"
    )

    new_frame = pd.DataFrame(
        rows,
        columns=RESULT_COLUMNS,
    )

    with lock_path.open("a+") as lock_file:
        fcntl.flock(
            lock_file.fileno(),
            fcntl.LOCK_EX,
        )

        if output_path.is_file():
            existing = pd.read_csv(
                output_path,
                dtype={
                    "task_lang": str,
                    "transfer_lang": str,
                },
                keep_default_na=False,
            )

            if list(existing.columns) != RESULT_COLUMNS:
                raise ValueError(
                    f"Unexpected columns in {output_path}: "
                    f"{list(existing.columns)}"
                )

            combined = pd.concat(
                [existing, new_frame],
                ignore_index=True,
            )
        else:
            combined = new_frame

        combined = (
            combined
            .drop_duplicates(
                subset=[
                    "task_lang",
                    "transfer_lang",
                ],
                keep="last",
            )
            .sort_values(
                [
                    "transfer_lang",
                    "task_lang",
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        temporary_path = output_path.with_name(
            f".{output_path.name}.{os.getpid()}.tmp"
        )

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                combined.to_csv(
                    handle,
                    index=False,
                )
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(
                temporary_path,
                output_path,
            )
        finally:
            temporary_path.unlink(
                missing_ok=True,
            )

        fcntl.flock(
            lock_file.fileno(),
            fcntl.LOCK_UN,
        )


def run_task(
    task: LlamaTask,
) -> None:
    args = build_parser(task).parse_args()

    if Path(args.lang).name != args.lang:
        raise ValueError(
            "--lang must be a language-folder name, not a path."
        )

    if args.eval_steps <= 0 or args.logging_steps <= 0:
        raise ValueError(
            "Step intervals must be positive."
        )

    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        raise ValueError(
            "Batch sizes must be positive."
        )

    local_rank = int(
        os.environ.get("LOCAL_RANK", "0")
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    compute_dtype = _resolve_dtype(
        args.precision
    )

    set_seed(args.seed)

    data_dir = args.data_root / task.name
    language_dir = data_dir / args.lang

    if not language_dir.is_dir():
        raise FileNotFoundError(
            "Transfer-language data not found: "
            f"{language_dir}"
        )

    run_dir = (
        args.model_root
        / task.name
        / "llama"
        / args.lang
    )
    adapter_dir = run_dir / "adapter"
    checkpoint_dir = run_dir / "checkpoints"

    logging_dir = (
        args.log_root
        / task.name
        / "llama"
        / args.lang
    )

    tokenizer = _load_tokenizer(
        args.base_model
    )

    data_collator = (
        task.data_collator_factory(tokenizer)
    )

    if args.eval_only:
        model = _load_adapter_model(
            task,
            args,
            tokenizer,
            compute_dtype,
            adapter_dir,
        )
    else:
        raw_datasets = load_from_disk(
            language_dir
        )

        for split in ("train", "validation"):
            if split not in raw_datasets:
                raise KeyError(
                    f"{language_dir} has no {split!r} split."
                )

        train_dataset = limit_dataset(
            raw_datasets["train"],
            args.max_train_samples,
        )
        validation_dataset = limit_dataset(
            raw_datasets["validation"],
            args.max_eval_samples,
        )

        train_dataset = task.preprocess_dataset(
            train_dataset,
            tokenizer,
            args,
        )
        validation_dataset = task.preprocess_dataset(
            validation_dataset,
            tokenizer,
            args,
        )

        model = _build_training_model(
            task,
            args,
            tokenizer,
            compute_dtype,
        )

        trainer = Trainer(
            model=model,
            args=_training_arguments(
                args,
                compute_dtype,
                checkpoint_dir,
                logging_dir,
            ),
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
            compute_metrics=task.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=(
                        args.patience
                    )
                )
            ],
        )

        trainer.train(
            resume_from_checkpoint=(
                args.resume_from_checkpoint
            )
        )

        adapter_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        trainer.save_model(
            str(adapter_dir)
        )
        tokenizer.save_pretrained(
            adapter_dir
        )

        model = trainer.model

        del (
            trainer,
            raw_datasets,
            train_dataset,
            validation_dataset,
        )

        gc.collect()
        torch.cuda.empty_cache()

    evaluator = Trainer(
        model=model,
        args=_prediction_arguments(
            args,
            compute_dtype,
            run_dir / "prediction",
        ),
        data_collator=data_collator,
        compute_metrics=task.compute_metrics,
    )

    available_languages = _dataset_languages(
        data_dir
    )
    task_languages = _validate_languages(
        args.task_langs,
        available_languages,
    )

    rows: list[dict[str, Any]] = []

    for task_language in task_languages:
        task_dataset = load_from_disk(
            data_dir / task_language
        )

        if "test" not in task_dataset:
            raise KeyError(
                f"{data_dir / task_language} "
                "has no 'test' split."
            )

        test_dataset = limit_dataset(
            task_dataset["test"],
            args.max_test_samples,
        )

        test_dataset = task.preprocess_dataset(
            test_dataset,
            tokenizer,
            args,
        )

        prediction = evaluator.predict(
            test_dataset,
            metric_key_prefix="test",
        )

        if "test_f1" not in prediction.metrics:
            raise RuntimeError(
                "compute_metrics did not return an F1 score."
            )

        score = float(
            prediction.metrics["test_f1"]
        )

        rows.append(
            {
                "task_lang": task_language,
                "transfer_lang": args.lang,
                "f1_score": score,
            }
        )

        print(
            f"task={task.name} "
            f"transfer_lang={args.lang} "
            f"task_lang={task_language} "
            f"f1={score:.6f}",
            flush=True,
        )

        del (
            task_dataset,
            test_dataset,
            prediction,
        )

        gc.collect()
        torch.cuda.empty_cache()

    output_path = (
        args.result_root
        / f"{task.name}_llama.csv"
    )

    _write_results(
        rows,
        output_path,
    )

    print(
        f"Updated {output_path}",
        flush=True,
    )