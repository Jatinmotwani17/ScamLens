# Novelty-bucketed metrics - `verdict_lr_plus`

> **This is the honest generalisation measure.** Each row is bucketed by `max_train_sim` - the maximum TF-IDF char-ngram cosine similarity to any TRAIN row (results/nn_sim.parquet). Performance that only holds at high similarity is near-copy matching, not detection.

Decisions use the fixed operating point (t_high = 0.039, temperature = 0.461) fitted on validation only.

## `test_b`

| max_train_sim | n | n scam | n safe | scam recall @t_high | FPR | AUROC | mean p | ECE | note |
|---|---:|---:|---:|---|---|---|---|---|---|
| 0.0-0.5 | 159 | 104 | 55 | 1.000 | 0.000 | 1.000 | 0.650 | 0.0058 |  |
| 0.5-0.7 | 55 | 27 | 28 | 1.000 | 0.036 | 1.000 | 0.496 | 0.0048 |  |
| 0.7-0.9 | 1 | 1 | 0 | 1.000 | n/a | n/a | 1.000 | 0.0001 | too small for claims; single class in bucket |

## `test_real/external_scam`

| max_train_sim | n | n scam | n safe | scam recall @t_high | FPR | AUROC | mean p | ECE | note |
|---|---:|---:|---:|---|---|---|---|---|---|
| 0.0-0.5 | 2,066 | 2,066 | 0 | 0.987 | n/a | n/a | 0.963 | 0.0372 | single class in bucket |
| 0.5-0.7 | 811 | 811 | 0 | 0.999 | n/a | n/a | 0.991 | 0.0090 | single class in bucket |
| 0.7-0.9 | 576 | 576 | 0 | 1.000 | n/a | n/a | 0.996 | 0.0039 | single class in bucket |
| >0.9 | 58 | 58 | 0 | 1.000 | n/a | n/a | 0.996 | 0.0042 | single class in bucket |

## `test_real/public_ham_english`

| max_train_sim | n | n scam | n safe | scam recall @t_high | FPR | AUROC | mean p | ECE | note |
|---|---:|---:|---:|---|---|---|---|---|---|
| 0.0-0.5 | 312 | 0 | 312 | n/a | 0.064 | n/a | 0.015 | 0.0154 | single class in bucket |
| 0.5-0.7 | 29 | 0 | 29 | n/a | 0.069 | n/a | 0.011 | 0.0111 | too small for claims; single class in bucket |
| 0.7-0.9 | 3 | 0 | 3 | n/a | 0.000 | n/a | 0.000 | 0.0001 | too small for claims; single class in bucket |
| >0.9 | 1 | 0 | 1 | n/a | 0.000 | n/a | 0.000 | 0.0001 | too small for claims; single class in bucket |

## `test_real/user_real_scam`

| max_train_sim | n | n scam | n safe | scam recall @t_high | FPR | AUROC | mean p | ECE | note |
|---|---:|---:|---:|---|---|---|---|---|---|
| 0.0-0.5 | 5 | 5 | 0 | 1.000 | n/a | n/a | 0.986 | 0.0139 | too small for claims; single class in bucket |

## `test_real/user_real_safe`

| max_train_sim | n | n scam | n safe | scam recall @t_high | FPR | AUROC | mean p | ECE | note |
|---|---:|---:|---:|---|---|---|---|---|---|
| 0.0-0.5 | 5 | 0 | 5 | n/a | 0.600 | n/a | 0.507 | 0.5065 | too small for claims; single class in bucket |

## `val`

| max_train_sim | n | n scam | n safe | scam recall @t_high | FPR | AUROC | mean p | ECE | note |
|---|---:|---:|---:|---|---|---|---|---|---|
| 0.0-0.5 | 2,809 | 2,191 | 618 | 0.986 | 0.050 | 0.997 | 0.750 | 0.0303 |  |
| 0.5-0.7 | 1,113 | 963 | 150 | 0.999 | 0.027 | 0.999 | 0.858 | 0.0075 |  |
| 0.7-0.9 | 569 | 552 | 17 | 1.000 | 0.059 | 1.000 | 0.960 | 0.0099 |  |
| >0.9 | 49 | 49 | 0 | 1.000 | n/a | n/a | 0.997 | 0.0032 | single class in bucket |

> AUROC needs both classes in a bucket; single-class buckets report recall/FPR only. Buckets under 30 rows are printed but are not claims.
