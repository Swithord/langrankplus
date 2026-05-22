# LangRankPlus

LangRankPlus implements source-language selection methods for cross-lingual transfer.

Given a target language and candidate source languages, the goal is to rank sources using language-distance features and select a good transfer source.

Each input row corresponds to:

```text
(target language, source language)
````

and contains:

1. a downstream transfer score, such as `accuracy` or `f1_score`;
2. distance features, such as `new_gen`, `new_typ`, `new_geo`, and `script`.

---

## Method families

| Family                            | Uses transfer labels for fitting? | What is fitted?                           | Examples                                   |
| --------------------------------- | --------------------------------: | ----------------------------------------- | ------------------------------------------ |
| Fully fixed rules                 |                                No | Nothing                                   | `single_*`, `composite_equal`, `rrf_equal` |
| Fitted fixed-form rankers         |                               Yes | Simple pairwise-trained weights / `rrf_k` | `composite_pairwise_*`, `rrf_pairwise_*`   |
| Supervised LangRank-style rankers |                               Yes | Flexible ranking model                    | `lightgbm_lambdarank`, `mlp_listnet`       |

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

## Near-oracle source-language selection

Exact top-1 source recovery is often too strict. Several source languages may give nearly indistinguishable transfer performance. LangRankPlus therefore also reports two operational near-oracle shortlist metrics.

For a target language `q`, define the oracle-best transfer score as

```text
M_q = max_s Y_{q,s}.
```

For tolerance `epsilon`, define the near-oracle source set as

```text
G_q(epsilon) = {s : Y_{q,s} >= (1 - epsilon) M_q}.
```

By default, the code uses

```text
epsilon = 0.05
```

so a near-oracle source is one whose transfer performance is within 5% of the oracle-best source.

### 1. CNOTC-90@5%

The main set-valued metric is:

```text
CNOTC-90@5%
```

This means **Conformal Near-Oracle Trial Complexity** at 90% coverage and 5% oracle tolerance.

For a ranking method `h`, let

```text
R_q^epsilon(h) = rank of the first source in G_q(epsilon) under h.
```

The CNOTC value is the split-conformal 90% quantile of these first-near-oracle ranks. Equivalently, it asks:

> How many source languages must I try to have a calibrated 90% chance of including at least one source within 5% of oracle performance?

A smaller value is better. It means the ranker needs fewer transfer trials to find a near-oracle source.

The default settings are:

```text
alpha = 0.1
epsilon = 0.05
```

so the default CNOTC metric is:

```text
CNOTC-90@5%
```

The summary output reports:

```text
cnotc_trial_complexity
cnotc_pool_fraction
cnotc_near_oracle_coverage
cnotc_exact_best_coverage
cnotc_best_in_set_performance_loss
```

Interpretation:

* `cnotc_trial_complexity`: average number of sources selected by the conformal near-oracle rule;
* `cnotc_pool_fraction`: selected fraction of the candidate pool;
* `cnotc_near_oracle_coverage`: empirical probability that the set contains a 5%-near-oracle source;
* `cnotc_exact_best_coverage`: empirical probability that the set contains an exact oracle-best source;
* `cnotc_best_in_set_performance_loss`: oracle regret after trying all sources in the returned set.

### 2. Budget-B@5%

The second operational metric fixes a practical shortlist budget:

```text
Budget-B@5%
```

For example, with the default

```text
B = 10
```

the evaluator returns the top 10 sources and checks whether that fixed-budget shortlist contains a 5%-near-oracle source.

This answers:

> If I can only try B source languages, how often does the shortlist contain a near-oracle source?

The default budget is:

```text
B = 10
```

and the summary output reports:

```text
budget_10_at_5_size
budget_10_at_5_pool_fraction
budget_10_at_5_near_oracle_coverage
budget_10_at_5_exact_best_coverage
budget_10_at_5_best_in_set_performance_loss
```

Additional budgets can be reported with:

```bash
--budget_ks 3 5 10
```

The CNOTC metric is the calibrated trial-complexity view. The budget metric is the fixed-deployment-budget view.

---

## Output files

Evaluation writes:

```text
summary_<performance_col>.csv
per_fold_<performance_col>.csv
pairwise_<performance_col>.csv
```

The summary file contains one row per method. The per-fold file contains one row per held-out query and method. The pairwise file contains paired method comparisons.

When CNOTC is enabled, the summary includes both:

```text
cnotc_*
budget_<B>_at_5_*
```

columns.

---

## Main same-benchmark evaluation

Run the full SIB200 comparison:

```bash
python main.py evaluate \
  --csv data/sib200.csv \
  --performance_col accuracy \
  --features new_gen new_typ new_geo script \
  --include_nested_offline \
  --include_cnotc \
  --cnotc_alpha 0.1 \
  --cnotc_epsilon 0.05 \
  --cnotc_cal_size 0.2 \
  --budget_ks 10 \
  --n_opt_steps 1000 \
  --learning_rate 0.05 \
  --max_pairs_per_query 5000 \
  --score_scale 10.0 \
  --val_size 0 \
  --outdir artifacts/evaluation_sib200_nested
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
  --include_cnotc \
  --cnotc_alpha 0.1 \
  --cnotc_epsilon 0.05 \
  --cnotc_cal_size 0.2 \
  --budget_ks 10 \
  --n_opt_steps 1000 \
  --learning_rate 0.05 \
  --max_pairs_per_query 5000 \
  --score_scale 10.0 \
  --val_size 0 \
  --outdir artifacts/evaluation_sib200_nested_f1
```

To also report budgets 3 and 5:

```bash
python main.py evaluate \
  --csv data/sib200.csv \
  --performance_col f1_score \
  --features new_gen new_typ new_geo script \
  --include_nested_offline \
  --include_cnotc \
  --budget_ks 3 5 10 \
  --outdir artifacts/evaluation_sib200_nested_f1
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
  --include_cnotc \
  --cnotc_alpha 0.1 \
  --cnotc_epsilon 0.05 \
  --cnotc_cal_size 0.2 \
  --budget_ks 10 \
  --val_size 0 \
  --outdir artifacts/evaluation_new_dataset_frozen
```

---

## Optional IR metrics

The default evaluation keeps the output parsimonious. It reports the main point-selection metrics plus CNOTC and the budgeted near-oracle shortlist metrics.

Additional IR-style metrics can be enabled with:

```bash
--include_ir_metrics
```

and cutoffs can be chosen with:

```bash
--ir_cutoffs 1 3 5 10
```

These are useful for diagnostic analyses, but the main operational metrics are CNOTC-90@5% and Budget-B@5%.

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

CNOTC and Budget-B@5% require labelled held-out or calibration queries, so they are evaluation metrics rather than pure inference outputs.

---

## Slurm

The main same-benchmark Slurm script is:

```bash
sbatch slurm/run_eval_nested.slurm
```

It runs nested pairwise composite/RRF, supervised rankers, fixed baselines, CNOTC-90@5%, and Budget-10@5% by default.

The faster diagnostic script is:

```bash
sbatch slurm/run_eval.slurm
```

The frozen-ranker evaluation script is:

```bash
sbatch slurm/run_eval_frozen.slurm
```

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

For operational source-language selection, the key quantities are:

```text
CNOTC-90@5%:
    how many source languages are needed for calibrated 90% near-oracle coverage

Budget-B@5%:
    how much near-oracle coverage is achieved under a fixed trial budget B
```

A useful method should have low CNOTC, high Budget-B@5% near-oracle coverage, and low best-in-set performance loss.

```

