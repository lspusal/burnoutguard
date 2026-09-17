#!/usr/bin/env python3
"""Signals available at an early horizon beyond the VLE clickstream.

The engagement features alone describe how much a student clicks, but by week 4 a
course has usually set deadlines, and failing to submit is one of the oldest known
precursors of withdrawal. Enrolment context (late registration, previous attempts,
credit load) and demographics are known from day zero. All of these are available
at week w, so using them introduces no look-ahead: assessment variables are
computed only from submissions made on or before day 7w, and the set of deadlines
counted is the set already due by that day.
"""
import numpy as np, pandas as pd
from pathlib import Path

KEYS = ["id_student", "code_module", "code_presentation"]
CATS = ["gender", "age_band", "imd_band", "disability", "highest_education"]


def add_horizon_features(f, week, data_dir="data", include_demographics=False):
    """Return (f_with_extra_columns, list_of_added_column_names).

    Protected attributes are merged in so that the fairness audit can group by
    them, but by default they are NOT returned as model inputs: an early warning
    system that flags students on the basis of prior education, gender, age,
    deprivation or disability is hard to defend whatever it buys, and here it buys
    about 0.01 AUROC. Set include_demographics=True to reproduce that variant.
    """
    md, d = week * 7, Path(data_dir)
    sa = pd.read_csv(d / "studentAssessment.csv")
    asm = pd.read_csv(d / "assessments.csv")
    reg = pd.read_csv(d / "studentRegistration.csv")
    info = pd.read_csv(d / "studentInfo.csv")

    a = sa.merge(asm, on="id_assessment")
    a = a[a["date_submitted"] <= md]                    # sólo lo entregado ya
    g = a.groupby(KEYS).agg(n_submitted=("score", "size"), mean_score=("score", "mean"),
                            min_score=("score", "min"),
                            last_submit=("date_submitted", "max")).reset_index()
    a["late"] = a["date_submitted"] > a["date"]
    late = a.groupby(KEYS)["late"].mean().reset_index(name="late_rate")
    due = (asm[asm["date"] <= md].groupby(["code_module", "code_presentation"])
           .size().reset_index(name="n_due"))                 # plazos ya vencidos

    f = (f.merge(g, on=KEYS, how="left").merge(late, on=KEYS, how="left")
          .merge(due, on=["code_module", "code_presentation"], how="left"))
    f["n_due"] = f["n_due"].fillna(0)
    f["n_submitted"] = f["n_submitted"].fillna(0)
    f["n_missed"] = (f["n_due"] - f["n_submitted"]).clip(lower=0)
    f["submit_rate"] = (f["n_submitted"] / f["n_due"].replace(0, np.nan)).fillna(0)
    f["days_since_submit"] = (md - f["last_submit"]).fillna(md)
    for c in ["mean_score", "min_score", "late_rate"]:
        f[c] = f[c].fillna(0)

    f = f.merge(reg[KEYS + ["date_registration"]], on=KEYS, how="left")
    f["date_registration"] = f["date_registration"].fillna(0)

    static = info[KEYS + ["num_of_prev_attempts", "studied_credits"] + CATS]
    f = f.merge(static, on=KEYS, how="left", suffixes=("", "_dup"))
    if include_demographics:
        for c in CATS:
            col = c if c in f.columns else c + "_dup"
            f[c + "_code"] = pd.factorize(f[col].astype(str))[0]

    added = ["n_submitted", "mean_score", "min_score", "late_rate", "n_due", "n_missed",
             "submit_rate", "days_since_submit", "date_registration",
             "num_of_prev_attempts", "studied_credits"]
    if include_demographics:
        added += [c + "_code" for c in CATS]
    return f, added
