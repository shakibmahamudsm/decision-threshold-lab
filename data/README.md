# Data

## What is committed here

`data/sample/` holds the **first 200 rows** of the R2 feature set and its labels,
gzipped. They are here as a **format reference only** — so you can see exactly
what the input looks like without downloading 200 MB.

| File | Shape | Notes |
|---|---|---|
| `R2_features_sample.csv.gz` | 200 × 768 | float32, header is `0,1,...,767` |
| `labels_sample.csv.gz` | 200 × 19 | 0/1, header is `0,1,...,18` |

**Do not train on this sample.** 200 rows is nowhere near enough — the rarest
label appears in 0.25% of the data, so this sample contains roughly zero
examples of it. It exists to document the schema, nothing more.

## What the app actually uses

The app does not read these CSVs at all. It reads the `.npy` probability
matrices in `assets/`, which are committed and are the output of
`prepare_data.py` run once on the **full** dataset.

That is the whole design: train offline, ship the predictions, keep the app
instant.

## The full dataset

25,000 rows × 768 features, plus 25,000 × 19 labels. It comes from a university
assignment and is deliberately anonymised — the features have no stated meaning
and the labels have no names.

To regenerate `assets/` yourself you need `R2_train.csv` and `labels_train.csv`
in one folder, then:

```bash
pip install -r requirements-dev.txt
python prepare_data.py --data-dir "path/to/that/folder"
```

Expect the SVM step to take roughly 30–60 minutes: it is a calibrated SVC fit
once per label, 19 times.

## Label distribution (full 25,000 rows)

Every label is a minority. This is the entire reason both of these projects
exist.

| Label | Positives | % | Label | Positives | % |
|---|---|---|---|---|---|
| 14 | 62 | 0.25% | 18 | 3,562 | 14.25% |
| 16 | 79 | 0.32% | 17 | 4,122 | 16.49% |
| 1 | 547 | 2.19% | 4 | 4,265 | 17.06% |
| 11 | 611 | 2.44% | 5 | 4,585 | 18.34% |
| 12 | 707 | 2.83% | 6 | 6,205 | 24.82% |
| 7 | 1,245 | 4.98% | 8 | 6,357 | 25.43% |
| 15 | 1,263 | 5.05% | 13 | 7,280 | 29.12% |
| 3 | 1,269 | 5.08% | 2 | 8,730 | 34.92% |
| 0 | 3,355 | 13.42% | 9 | 9,025 | 36.10% |
| | | | 10 | 9,289 | 37.16% |

Labels **1, 11, 12, 14 and 16** are the five minority classes. Rows carry 2.9
labels on average, and only 5 rows out of 25,000 have no label at all.
