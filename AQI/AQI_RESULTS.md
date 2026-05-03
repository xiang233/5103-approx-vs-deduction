# AQI Case Study

This document summarizes the AQI experiment added for the project proposal, explains how to run it, and records the preprocessing methodology used to turn the raw AQI data into the result CSV files.

## What This Experiment Does

The AQI case study follows the same approximate-reasoning pattern as the mushroom case study, but on environmental data.

Input data:

- `AQI/aqi_daily_1980_to_2021.csv` (https://www.kaggle.com/datasets/threnjen/40-years-of-air-quality-index-from-the-epa-daily)

Implemented code:

- [src/aqi_booleanize.py](/src/aqi_booleanize.py)
- [src/experiments/exp5_aqi_case.py](/src/experiments/exp5_aqi_case.py)

Generated outputs:

- [results/exp5_aqi_proposal_rules.csv](/results/exp5_aqi_proposal_rules.csv)
- [results/exp5_aqi_ozone_top_conjs.csv](/results/exp5_aqi_ozone_top_conjs.csv)
- [results/exp5_aqi_pm25_top_conjs.csv](/results/exp5_aqi_pm25_top_conjs.csv)
- [results/exp5_aqi_ozone_delta_gamma.csv](/results/exp5_aqi_ozone_delta_gamma.csv)
- [results/exp5_aqi_pm25_delta_gamma.csv](/results/exp5_aqi_pm25_delta_gamma.csv)

## Methodology

### 1. Raw data source

The raw AQI file is the EPA-style daily AQI table:

- `State Name`
- `Date`
- `AQI`
- `Category`
- `Defining Parameter`
- `Latitude`
- `Longitude`
- `County Name`

Each row is a county-day observation.

### 2. Row filtering

The AQI experiment keeps only rows that satisfy both conditions:

- year is between `2016` and `2021`
- `AQI >= 51`

This means the experiment focuses on elevated-AQI days rather than all days, which makes the pollutant-dominance rules more meaningful.

### 3. Boolean feature extraction

Each retained county-day is converted into a short Boolean feature vector with interpretable context predicates:

- season: `season=winter`, `season=spring`, `season=summer`, `season=fall`
- broad region: `region=West`, `region=Midwest`, `region=South`, `region=Northeast`, `region=Territory`
- coastal proxy: `coastal=True`, `coastal=False`

The transformation is implemented in [src/aqi_booleanize.py](/src/aqi_booleanize.py).

Season is derived from the month in `Date`.

Region is derived from `State Name` using fixed US-region groupings.

`coastal=True` is a simple state-level proxy based on whether the state is coastal.

### 4. Target-label extraction

The same filtered AQI rows are reused for two separate binary prediction targets:

- `target_parameter="Ozone"`
- `target_parameter="PM2.5"`

For each target, the label is:

- `y = 1` if `Defining Parameter` equals that target pollutant
- `y = 0` otherwise

So the experiment asks questions like:

- among elevated-AQI days, when is Ozone usually the defining pollutant?
- among elevated-AQI days, when is PM2.5 usually the defining pollutant?

### 5. Train/test split

The split is temporal instead of random:

- train: `2016-2019`
- test: `2020-2021`

This is implemented by `temporal_train_test_split(...)` in [src/aqi_booleanize.py](/src/aqi_booleanize.py).

That choice is important because it checks whether approximate rules still generalize on later years instead of just on a random holdout from the same period.

### 6. Result CSV extraction

The experiment writes three kinds of result CSVs from the processed AQI data:

1. `exp5_aqi_proposal_rules.csv`
   This scores a few hand-chosen rules inspired directly by the proposal, such as `summer -> Ozone` and `winter -> PM2.5`.

2. `exp5_aqi_*_top_conjs.csv`
   These files enumerate strong singleton and pairwise Boolean antecedents and score them on both train and test.

3. `exp5_aqi_*_delta_gamma.csv`
   These files measure context sensitivity using `Delta_gamma`, showing how a base rule changes after adding more context.

Each row in these output CSVs is computed from the Booleanized AQI dataset using:

- empirical validity
- coverage
- Wilson confidence intervals
- classical acceptance / counterexample count
- context sensitivity when applicable

## Main Results

### Dataset sizes

After filtering to `2016-2021` and `AQI >= 51`:

- total elevated-AQI rows: `295,839`
- train rows: `241,968`
- test rows: `53,871`

Positive label rates:

- Ozone: train `0.4430`, test `0.3127`
- PM2.5: train `0.4914`, test `0.6101`

### Proposal-style rules

From [results/exp5_aqi_proposal_rules.csv](/results/exp5_aqi_proposal_rules.csv):

| Rule | Split | Validity | Coverage | n_alpha |
|---|---|---:|---:|---:|
| `summer -> Ozone [AQI>=51]` | train | 0.6310 | 0.3443 | 83,309 |
| `summer -> Ozone [AQI>=51]` | test | 0.6109 | 0.3239 | 17,450 |
| `West & summer -> Ozone [AQI>=51]` | train | 0.7035 | 0.1260 | 30,491 |
| `West & summer -> Ozone [AQI>=51]` | test | 0.6889 | 0.1067 | 5,747 |
| `winter -> PM2.5 [AQI>=51]` | train | 0.9070 | 0.2006 | 48,535 |
| `winter -> PM2.5 [AQI>=51]` | test | 0.9101 | 0.2614 | 14,082 |
| `inland & winter -> PM2.5 [AQI>=51]` | train | 0.8947 | 0.1049 | 25,384 |
| `inland & winter -> PM2.5 [AQI>=51]` | test | 0.8942 | 0.1421 | 7,653 |

Interpretation:

- Ozone is strongly associated with summer elevated-AQI days, especially in the West.
- PM2.5 is very strongly associated with winter elevated-AQI days.
- None of these rules are classically valid, but they are still highly useful under approximate validity.

### Best discovered Ozone rules

Top generalizing Ozone rules from [results/exp5_aqi_ozone_top_conjs.csv](/results/exp5_aqi_ozone_top_conjs.csv):

| Antecedent | Train validity | Test validity | Test n_alpha |
|---|---:|---:|---:|
| `season=summer & region=Midwest` | 0.6541 | 0.7069 | 4,797 |
| `season=spring & region=West` | 0.7155 | 0.7068 | 2,756 |
| `season=summer & region=West` | 0.7035 | 0.6889 | 5,747 |
| `season=summer & coastal=False` | 0.6592 | 0.6846 | 10,231 |
| `season=summer` | 0.6310 | 0.6109 | 17,450 |

### Best discovered PM2.5 rules

Top generalizing PM2.5 rules from [results/exp5_aqi_pm25_top_conjs.csv](/results/exp5_aqi_pm25_top_conjs.csv):

| Antecedent | Train validity | Test validity | Test n_alpha |
|---|---:|---:|---:|
| `season=winter & region=Midwest` | 0.9402 | 0.9374 | 4,151 |
| `season=winter & coastal=True` | 0.9205 | 0.9291 | 6,429 |
| `season=winter & region=South` | 0.8935 | 0.9138 | 2,913 |
| `season=winter` | 0.9070 | 0.9101 | 14,082 |
| `season=winter & coastal=False` | 0.8947 | 0.8942 | 7,653 |

### Context sensitivity results

From [results/exp5_aqi_ozone_delta_gamma.csv](/results/exp5_aqi_ozone_delta_gamma.csv):

- base rule `region=West -> Ozone` has test validity `0.3489`
- refining with `season=summer` increases test validity by `+0.3400`
- refining with `season=spring` increases test validity by `+0.3579`
- refining with `season=winter` decreases test validity by `-0.3165`

From [results/exp5_aqi_pm25_delta_gamma.csv](/results/exp5_aqi_pm25_delta_gamma.csv):

- base rule `coastal=True -> PM2.5` has test validity `0.6769`
- refining with `season=winter` increases test validity by `+0.2522`
- refining with `season=fall` increases test validity by `+0.1087`
- refining with `season=spring` decreases test validity by `-0.1859`
- refining with `season=summer` decreases test validity by `-0.2157`

Interpretation:

- AQI pollutant dominance is clearly context-sensitive.
- The same coarse subgroup can support or weaken a rule depending on season.
- This is exactly the kind of approximate, non-monotonic behavior described in the proposal.

## How To Run

### 1. Run the AQI experiment

From the repository root:

```bash
python3 -m src.experiments.exp5_aqi_case
```

This regenerates:

- `results/exp5_aqi_proposal_rules.csv`
- `results/exp5_aqi_ozone_top_conjs.csv`
- `results/exp5_aqi_pm25_top_conjs.csv`
- `results/exp5_aqi_ozone_delta_gamma.csv`
- `results/exp5_aqi_pm25_delta_gamma.csv`

### 2. Run the smoke tests

```bash
python3 tests/test_smoke.py
```

This includes a targeted test for AQI preprocessing and temporal splitting.

### 3. Optional: run the mushroom experiment for comparison

```bash
python3 -m src.experiments.exp4_mushroom_case
```

## Notes

- The AQI case study was shaped by the proposal PDF and follows the existing mushroom experiment style.
- The experiment intentionally uses a small, interpretable Boolean vocabulary rather than a large feature engineering pipeline.
- The result CSVs are meant to be cited directly in the final writeup or presentation.
