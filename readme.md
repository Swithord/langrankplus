# LangRankPlus

LangRankPlus implements source-language selection methods for cross-lingual transfer.

Given a target language and candidate source languages, the goal is to rank sources using language-distance features and select a good transfer source.

Each input row corresponds to:

```text
(target language, source language)
```

and contains:

1. a downstream transfer score, such as `accuracy` or `f1_score`;
2. distance features, such as `new_gen`, `new_typ`, `new_geo`, and `script`.

---

## Method families

| Family | Uses transfer labels for fitting? | What is fitted? | Examples |
|---|---:|---|---|
| Fully fixed rules | No | Nothing | `single_*`, `composite_equal`, `rrf_equal` |
| Fitted fixed-form rankers | Yes | Simple pairwise-trained weights / `rrf_k` | `composite_pairwise_*`, `rrf_pairwise_*` |
| Supervised LangRank-style rankers | Yes | Flexible ranking model | `lightgbm_lambdarank`, `mlp_listnet` |

In nested evaluation, fitted composite/RRF and supervised LangRank-style rankers use the same fitting queries. They differ in model capacity:

```text
composite/RRF:
    constrained weighted distance or rank-fusion rule

LightGBM/MLP:
    higher-capacity supervised ranking model
```

---

## Repository structure

```text
langrankplus/
├── data/
│   └── sib200.csv
├── main.py
├── requirements.txt
├── scripts/
│   ├── train_offline.py
│   ├── evaluate_methods.py
│   └── infer.py
├── slurm/
│   ├── README.md
│   ├── run_eval.slurm
│   ├── run_eval_nested.slurm
│   ├── run_train_offline.slurm
│   ├── run_eval_frozen.slurm
│   └── run_infer.slurm
└── src/
    ├── data.py
    ├── evaluation.py
    ├── metrics.py
    ├── validation.py
    ├── conformal.py
    └── rankers/
        ├── base.py
        ├── single.py
        ├── composite.py
        ├── rrf.py
        ├── random.py
        ├── lightgbm.py
        └── mlp.py
```

---

## Data format

Each CSV should contain one row per target-source pair:

```text
task_lang,transfer_lang,accuracy,f1_score,new_gen,new_typ,new_geo,script
```

where:

* `task_lang` is the target language;
* `transfer_lang` is the candidate source language;
* `accuracy` and `f1_score` are downstream transfer scores;
* `new_gen`, `new_typ`, `new_geo`, and `script` are distance features.

Lower distance values are assumed to mean closer languages and better candidate sources.

For multiple datasets, pass multiple CSVs. The loader adds a `dataset` column using each file stem unless explicit dataset names are provided.

---

## Methods

### Fully fixed rules

```text
random
single_new_gen
single_new_typ
single_new_geo
single_script
composite_equal
rrf_equal
```

These methods do not use transfer-performance labels.

`single_*` ranks by one distance feature.  
`composite_equal` averages normalized distances.  
`rrf_equal` combines within-query feature ranks using reciprocal rank fusion.

### Fitted fixed-form rankers

```text
composite_pairwise_nested
rrf_pairwise_nested
composite_pairwise_frozen
rrf_pairwise_frozen
```

`composite_pairwise_*` fits nonnegative weights in

```text
score(target, source) = - sum_m w_m distance_m(target, source),
```

with

```text
w_m >= 0
sum_m w_m = 1.
```

`rrf_pairwise_*` fits weights and selects `rrf_k` in

```text
RRF(source) = sum_m w_m / (rrf_k + rank_m(source)).
```

Both are trained with a pairwise ranking surrogate over sources within the same target query.

### Supervised LangRank-style rankers

```text
lightgbm_lambdarank
mlp_listnet
```

These train supervised ranking models from labelled transfer results.

---

## Evaluation

The evaluator uses leave-one-query-out cross-validation.

A query is:

```text
task_lang
```

for a single dataset, or

```text
(dataset, task_lang)
```

for multi-dataset evaluation.

For each held-out query:

```text
1. Remove the held-out query.
2. Fit pairwise composite/RRF and supervised rankers on the same fitting queries.
3. Score all candidate sources for the held-out query.
4. Select the source with the highest predicted score.
5. Evaluate using the observed held-out transfer scores.
```

The main point-selection metrics are:

```text
NDCG@k
top-1 performance loss
top-1 accuracy
top-3 accuracy
```

Top-1 performance loss is

```text
(best_observed_performance - selected_source_performance)
    / best_observed_performance.
```

A value of `0` means the method selected an optimal source.

---

## Conformal source-language selection

Conformal prediction adds a set-valued layer on top of any ranker.

Instead of returning one source,

```text
selected source = argmax_source score(target, source),
```

it returns:

```text
C(target) = {sources whose scores are close enough to the top score}.
```

The target guarantee is best-source coverage:

```text
P(oracle-best source is in C(target)) >= 1 - alpha.
```

For example, `alpha=0.1` targets 90% coverage.

The conformal unit is the target-language query, not an individual target-source row.

When conformal evaluation is enabled, each outer fold uses:

```text
fitting queries:
    fit composite/RRF or train supervised rankers

conformal-calibration queries:
    estimate the conformal threshold

held-out query:
    evaluate point prediction and conformal source set
```

Conformal output metrics include:

```text
conformal_best_source_coverage
conformal_average_set_size
conformal_singleton_rate
conformal_empty_rate
conformal_best_in_set_performance_loss
```

These should be read as a coverage-size tradeoff.

---

## Output files

Evaluation writes:

```text
summary_<performance_col>.csv
per_fold_<performance_col>.csv
pairwise_<performance_col>.csv
```

The summary file contains one row per method. The per-fold file contains one row per held-out query and method. The pairwise file contains paired method comparisons.

---

## Main same-benchmark evaluation

Run the full SIB200 comparison:

```bash
python main.py evaluate \
  --csv data/sib200.csv \
  --performance_col accuracy \
  --features new_gen new_typ new_geo script \
  --include_nested_offline \
  --include_conformal \
  --conformal_alpha 0.1 \
  --conformal_cal_size 0.2 \
  --n_opt_steps 1000 \
  --learning_rate 0.05 \
  --max_pairs_per_query 5000 \
  --score_scale 10.0 \
  --val_size 0 \
  --outdir artifacts/evaluation_sib200_nested_conformal
```

This compares:

```text
random
single_new_gen
single_new_typ
single_new_geo
single_script
composite_equal
rrf_equal
composite_pairwise_nested
rrf_pairwise_nested
lightgbm_lambdarank
mlp_listnet
```

For F1 instead of accuracy:

```bash
python main.py evaluate \
  --csv data/sib200.csv \
  --performance_col f1_score \
  --features new_gen new_typ new_geo script \
  --include_nested_offline \
  --include_conformal \
  --conformal_alpha 0.1 \
  --conformal_cal_size 0.2 \
  --n_opt_steps 1000 \
  --val_size 0 \
  --outdir artifacts/evaluation_sib200_nested_conformal_f1
```

---

## Fit frozen rankers

Fit pairwise composite/RRF rankers for frozen evaluation or inference:

```bash
python main.py train \
  --csv data/sib200.csv \
  --performance_col accuracy \
  --features new_gen new_typ new_geo script \
  --n_opt_steps 5000 \
  --learning_rate 0.05 \
  --max_pairs_per_query 5000 \
  --score_scale 10.0 \
  --outdir artifacts/offline_calibration_sib200
```

This writes:

```text
artifacts/offline_calibration_sib200/composite_accuracy.json
artifacts/offline_calibration_sib200/rrf_accuracy.json
```

For multiple historical datasets:

```bash
python main.py train \
  --csv data/sib200.csv data/masakhaner.csv data/tydiqa.csv \
  --performance_col accuracy \
  --features new_gen new_typ new_geo script \
  --outdir artifacts/offline_calibration_multi_dataset
```

---

## Evaluate frozen rankers

Use frozen fitted rankers when fitting data and evaluation data are separate:

```bash
python main.py evaluate \
  --csv data/new_dataset.csv \
  --performance_col accuracy \
  --features new_gen new_typ new_geo script \
  --composite_calibration_json artifacts/offline_calibration_sib200/composite_accuracy.json \
  --rrf_calibration_json artifacts/offline_calibration_sib200/rrf_accuracy.json \
  --include_conformal \
  --conformal_alpha 0.1 \
  --conformal_cal_size 0.2 \
  --val_size 0 \
  --outdir artifacts/evaluation_new_dataset_frozen
```

---

## Online inference

If a new CSV has distance features but no transfer labels:

```bash
python main.py infer \
  --csv data/new_task.csv \
  --features new_gen new_typ new_geo script \
  --method composite_equal \
  --outdir artifacts/inference_new_task
```

For a fitted selector:

```bash
python main.py infer \
  --csv data/new_task.csv \
  --features new_gen new_typ new_geo script \
  --method calibrated \
  --calibration_json artifacts/offline_calibration_multi_dataset/rrf_accuracy.json \
  --outdir artifacts/inference_new_task_calibrated
```

Conformal prediction is not used in pure inference unless labelled calibration queries are available.

---

## Slurm

The main same-benchmark Slurm script is:

```bash
sbatch slurm/run_eval_nested.slurm
```

It runs nested pairwise composite/RRF, supervised rankers, fixed baselines, and conformal evaluation.

See:

```text
slurm/README.md
```

for the Slurm workflow.

---

## Interpreting results

Use nested results for same-benchmark comparisons:

```text
composite_pairwise_nested
rrf_pairwise_nested
lightgbm_lambdarank
mlp_listnet
```

These methods use the same fitting queries and differ in model capacity.

Use frozen results for cross-dataset or deployment-style evaluation:

```text
composite_pairwise_frozen
rrf_pairwise_frozen
```

Conformal results should be interpreted as a coverage-size tradeoff. High coverage is useful only if the source sets are reasonably small.