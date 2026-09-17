#!/usr/bin/env python3
"""Temporal risk set for the early-prediction horizons.

At week w an early warning system can only act on students who are still
enrolled. OULAD records the day a student unregistered in
studentRegistration.date_unregistration; anyone whose unregistration day falls on
or before day 7w has already withdrawn by the time the prediction is made, so the
outcome is no longer predictable in any useful sense and no intervention is
possible. Those rows are removed from the sample at that horizon.

Keeping them inflates early-prediction performance substantially, because their
activity has already stopped and they are trivially separable.
"""
import pandas as pd
from pathlib import Path

KEYS = ["id_student", "code_module", "code_presentation"]


def apply_risk_set(df, horizon_day, data_dir="data"):
    """Drop students already unregistered on or before `horizon_day`."""
    reg = pd.read_csv(Path(data_dir) / "studentRegistration.csv")[KEYS + ["date_unregistration"]]
    m = df.merge(reg, on=KEYS, how="left")
    keep = m["date_unregistration"].isna() | (m["date_unregistration"] > horizon_day)
    return m[keep].drop(columns=["date_unregistration"]).reset_index(drop=True)
