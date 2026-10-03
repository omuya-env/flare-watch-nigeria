"""Shared cached loaders so every page reads the processed CSVs in one place."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def _read_csv(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_sites() -> pd.DataFrame:
    return _read_csv("sites.csv")


@st.cache_data(show_spinner=False)
def load_panel() -> pd.DataFrame:
    return _read_csv("site_year_panel.csv")


@st.cache_data(show_spinner=False)
def load_co2e() -> pd.DataFrame:
    return _read_csv("site_year_co2e.csv")


@st.cache_data(show_spinner=False)
def load_anomalies() -> pd.DataFrame:
    return _read_csv("anomalies.csv")


@st.cache_data(show_spinner=False)
def load_site_risk() -> pd.DataFrame:
    return _read_csv("site_risk.csv")


@st.cache_data(show_spinner=False)
def load_yearly() -> pd.DataFrame:
    return _read_csv("national_yearly.csv")


@st.cache_data(show_spinner=False)
def load_national() -> dict:
    path = PROCESSED / "national_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}