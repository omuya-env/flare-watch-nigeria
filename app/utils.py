"""Shared cached loaders so every page reads the processed CSVs in one place."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def _stamp(name: str) -> float:
    """Key the cache on the file mtime so regenerated data appears without a restart."""
    path = PROCESSED / name
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def _read_csv(name: str, _modified: float) -> pd.DataFrame:
    path = PROCESSED / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _read_json(name: str, _modified: float) -> dict:
    path = PROCESSED / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_sites() -> pd.DataFrame:
    return _read_csv("sites.csv", _stamp("sites.csv"))


def load_panel() -> pd.DataFrame:
    return _read_csv("site_year_panel.csv", _stamp("site_year_panel.csv"))


def load_co2e() -> pd.DataFrame:
    return _read_csv("site_year_co2e.csv", _stamp("site_year_co2e.csv"))


def load_anomalies() -> pd.DataFrame:
    return _read_csv("anomalies.csv", _stamp("anomalies.csv"))


def load_site_risk() -> pd.DataFrame:
    return _read_csv("site_risk.csv", _stamp("site_risk.csv"))


def load_yearly() -> pd.DataFrame:
    return _read_csv("national_yearly.csv", _stamp("national_yearly.csv"))


def load_national() -> dict:
    return _read_json("national_summary.json", _stamp("national_summary.json"))