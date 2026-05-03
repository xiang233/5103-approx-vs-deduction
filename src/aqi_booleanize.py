"""Boolean preprocessing for the AQI case study in the project proposal.

Each example is a county-day with AQI >= `min_aqi`. The target label is whether
the day's defining pollutant matches `target_parameter`. Antecedents are
interpretable context predicates over season and broad geography so the
experiment can study contextual validity and non-monotonic effects.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_NAMES = [
    "season=winter",
    "season=spring",
    "season=summer",
    "season=fall",
    "region=West",
    "region=Midwest",
    "region=South",
    "region=Northeast",
    "region=Territory",
    "coastal=True",
    "coastal=False",
]


REGIONS = {
    "West": {
        "Alaska", "Arizona", "California", "Colorado", "Hawaii", "Idaho",
        "Montana", "Nevada", "New Mexico", "Oregon", "Utah", "Washington",
        "Wyoming",
    },
    "Midwest": {
        "Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota",
        "Missouri", "Nebraska", "North Dakota", "Ohio", "South Dakota",
        "Wisconsin",
    },
    "South": {
        "Alabama", "Arkansas", "Delaware", "District Of Columbia", "Florida",
        "Georgia", "Kentucky", "Louisiana", "Maryland", "Mississippi",
        "North Carolina", "Oklahoma", "South Carolina", "Tennessee", "Texas",
        "Virginia", "West Virginia",
    },
    "Northeast": {
        "Connecticut", "Maine", "Massachusetts", "New Hampshire",
        "New Jersey", "New York", "Pennsylvania", "Rhode Island", "Vermont",
    },
}


COASTAL_STATES = {
    "Alaska", "Alabama", "California", "Connecticut", "Delaware", "Florida",
    "Georgia", "Hawaii", "Louisiana", "Maine", "Maryland", "Massachusetts",
    "Mississippi", "New Hampshire", "New Jersey", "New York",
    "North Carolina", "Oregon", "Rhode Island", "South Carolina", "Texas",
    "Virginia", "Washington",
}


@dataclass(frozen=True)
class AQIDataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    name_to_col: dict[str, int]
    years: np.ndarray
    target_parameter: str
    min_aqi: int


def load_aqi_context_dataset(path: str | Path = "AQI/aqi_daily_1980_to_2021.csv",
                             target_parameter: str = "Ozone",
                             min_year: int = 2016,
                             max_year: int = 2021,
                             min_aqi: int = 51) -> AQIDataset:
    """Load elevated-AQI county-days as a Boolean context dataset.

    The feature vocabulary is intentionally small and interpretable:
    season, broad US region, and a coastal/inland proxy. This keeps the AQI
    case study aligned with the proposal's focus on subgroup-sensitive rules.
    """
    x_rows: list[list[bool]] = []
    y_rows: list[bool] = []
    years: list[int] = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row["Date"][:4])
            if year < min_year or year > max_year:
                continue
            if int(row["AQI"]) < min_aqi:
                continue
            x_rows.append(_feature_row(row))
            y_rows.append(row["Defining Parameter"] == target_parameter)
            years.append(year)

    n_features = len(FEATURE_NAMES)
    X = np.asarray(x_rows, dtype=bool) if x_rows else np.zeros((0, n_features), dtype=bool)
    y = np.asarray(y_rows, dtype=bool)
    year_arr = np.asarray(years, dtype=np.int16)
    return AQIDataset(
        X=X,
        y=y,
        feature_names=list(FEATURE_NAMES),
        name_to_col={name: i for i, name in enumerate(FEATURE_NAMES)},
        years=year_arr,
        target_parameter=target_parameter,
        min_aqi=min_aqi,
    )


def predicate_mask(ds: AQIDataset, name: str) -> np.ndarray:
    return ds.X[:, ds.name_to_col[name]]


def temporal_train_test_split(ds: AQIDataset,
                              test_start_year: int = 2020) -> tuple[AQIDataset, AQIDataset]:
    """Split by calendar year to test generalisation on later AQI data."""
    train_mask = ds.years < test_start_year
    test_mask = ~train_mask
    return _subset(ds, train_mask), _subset(ds, test_mask)


def _subset(ds: AQIDataset, mask: np.ndarray) -> AQIDataset:
    return AQIDataset(
        X=ds.X[mask],
        y=ds.y[mask],
        feature_names=ds.feature_names,
        name_to_col=ds.name_to_col,
        years=ds.years[mask],
        target_parameter=ds.target_parameter,
        min_aqi=ds.min_aqi,
    )


def _feature_row(row: dict[str, str]) -> list[bool]:
    season = _season_of_date(row["Date"])
    region = _region_of_state(row["State Name"])
    coastal = row["State Name"] in COASTAL_STATES
    return [
        season == "winter",
        season == "spring",
        season == "summer",
        season == "fall",
        region == "West",
        region == "Midwest",
        region == "South",
        region == "Northeast",
        region == "Territory",
        coastal,
        not coastal,
    ]


def _season_of_date(date_str: str) -> str:
    month = int(date_str[5:7])
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def _region_of_state(state_name: str) -> str:
    for region, states in REGIONS.items():
        if state_name in states:
            return region
    return "Territory"
