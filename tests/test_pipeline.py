"""Unit tests for the continuity rules, the conversions and the explanation builder."""

import ast

import numpy as np
import pandas as pd
import pytest

import config
import download_data
from build_timeseries import add_features
from estimate_co2e import homes_equivalent, to_co2e_tonnes, to_kwh_electricity


def sample_panel() -> pd.DataFrame:
    """One site that flares, goes dark for two years, relights and spikes, plus a quiet site."""
    return pd.DataFrame(
        {
            "id": ["a"] * 6 + ["b"] * 6,
            "year": [2020, 2021, 2022, 2023, 2024, 2025] * 2,
            "BCM": [0.10, 0.10, 0.00, 0.00, 0.45, 0.10, 0.02, 0.02, 0.00, 0.02, 0.02, 0.02],
            "active": [True, True, False, False, True, True, True, True, False, True, True, True],
            "detection_freq": [0.9, 0.9, np.nan, np.nan, 0.95, 0.9, 0.5, 0.5, np.nan, 0.5, 0.5, 0.5],
        }
    )


def test_panel_is_sorted_and_dense():
    panel = add_features(sample_panel())
    assert panel.groupby("id")["year"].diff().dropna().eq(1).all()


def test_relight_fires_after_a_dark_run():
    panel = add_features(sample_panel())
    site_a = panel[panel["id"] == "a"].set_index("year")
    assert bool(site_a.loc[2024, "relight"]) is True
    assert site_a.loc[2024, "gap_years"] == 2
    assert bool(site_a.loc[2021, "relight"]) is False


def test_gap_years_resets_once_the_site_relights():
    panel = add_features(sample_panel())
    site_a = panel[panel["id"] == "a"].set_index("year")
    assert site_a.loc[2025, "gap_years"] == 0


def test_shutdown_and_extended_shutdown():
    panel = add_features(sample_panel())
    site_a = panel[panel["id"] == "a"].set_index("year")
    assert bool(site_a.loc[2022, "shutdown"]) is True
    assert bool(site_a.loc[2023, "shutdown"]) is False
    assert bool(site_a.loc[2023, "extended_shutdown"]) is True


def test_extended_shutdown_fires_once_per_episode():
    panel = add_features(sample_panel())
    site_a = panel[panel["id"] == "a"]
    assert site_a["extended_shutdown"].sum() == 1


def test_new_site_not_flagged_in_first_record_year():
    years = list(range(config.YEAR_START, config.YEAR_START + 4))
    panel = pd.DataFrame(
        {
            "id": ["early"] * 4 + ["late"] * 4,
            "year": years * 2,
            "BCM": [0.01] * 4 + [0.0, 0.0, 0.0, 0.02],
            "active": [True] * 4 + [False, False, False, True],
            "detection_freq": [0.5] * 8,
        }
    )
    panel = add_features(panel)
    assert panel[panel["id"] == "early"]["new_site"].sum() == 0
    assert panel[panel["id"] == "late"]["new_site"].sum() == 1


def test_new_site_fires_once_per_site():
    panel = add_features(sample_panel())
    assert panel[panel["new_site"]]["year"].tolist() == [2020, 2020]


def test_volume_spike_compares_against_active_years_only():
    panel = add_features(sample_panel())
    site_a = panel[panel["id"] == "a"].set_index("year")
    assert bool(site_a.loc[2024, "volume_spike"]) is True
    assert bool(site_a.loc[2021, "volume_spike"]) is False


def test_short_dark_gap_still_counts_as_a_relight():
    panel = add_features(sample_panel())
    site_b = panel[panel["id"] == "b"].set_index("year")
    assert site_b.loc[2023, "gap_years"] == 1
    assert bool(site_b.loc[2023, "relight"]) is True
    assert bool(site_b.loc[2022, "extended_shutdown"]) is False


def test_co2e_factor_matches_config():
    assert to_co2e_tonnes(1.0) == pytest.approx(1e9 * config.CO2E_T_PER_M3)
    assert to_co2e_tonnes(0.0) == 0.0


def test_emission_factor_reproduces_world_bank_global_total():
    """148 bcm paired with 381 Mt CO2e in 2023 is the published global calibration."""
    total_mt = 148e9 * config.CO2E_T_PER_M3 / 1e6
    assert total_mt == pytest.approx(381, rel=0.02)


def test_kwh_and_homes_arithmetic():
    kwh = to_kwh_electricity(1e9)
    assert kwh == pytest.approx(1e9 * config.KWH_THERMAL_PER_M3 * config.PLANT_EFFICIENCY)
    assert homes_equivalent(1e9) == pytest.approx(kwh / config.HOUSEHOLD_KWH_PER_YEAR)


def test_already_present_guards_partial_downloads(tmp_path):
    complete = tmp_path / "done.xlsx"
    complete.write_bytes(b"x" * 500)
    partial = tmp_path / "small.xlsx"
    partial.write_bytes(b"x" * 5)
    assert download_data.already_present(complete, 100) is True
    assert download_data.already_present(partial, 100) is False
    assert download_data.already_present(tmp_path / "missing.xlsx", 100) is False


def test_manifest_lists_every_source():
    sources = download_data.build_sources()
    names = {source["name"] for source in sources}
    assert {"worldbank_sites", "worldbank_country", "boundaries"} <= names
    assert f"eog_{max(config.EOG_FILES)}" in names


def test_explanations_have_one_sentence_per_flagged_row():
    from train_anomaly import MODEL_FEATURES, explain_shap

    from sklearn.ensemble import IsolationForest

    rng = np.random.default_rng(0)
    matrix = pd.DataFrame(rng.normal(size=(200, len(MODEL_FEATURES))), columns=MODEL_FEATURES)
    typical = {column: float(matrix[column].median()) for column in MODEL_FEATURES}
    model = IsolationForest(n_estimators=20, random_state=0).fit(matrix.to_numpy())
    sentences = explain_shap(model, matrix, typical)
    assert len(sentences) == len(matrix)
    assert all(sentence.startswith("top drivers: ") for sentence in sentences)
    assert all(len(sentence.split(";")) <= 3 for sentence in sentences)


def test_describe_feature_uses_reader_units():
    from train_anomaly import MODEL_FEATURES, describe_feature

    assert "bcm" in describe_feature(MODEL_FEATURES.index("BCM"), 0.02, 0.005)
    assert "clear-sky passes" in describe_feature(MODEL_FEATURES.index("detection_freq"), 0.2, 0.6)
    assert "standard deviations above" in describe_feature(MODEL_FEATURES.index("volume_zscore"), 3.0, 0.0)


@pytest.mark.skipif(not config.ANOMALIES_CSV.exists(), reason="pipeline output not built yet")
def test_every_reported_spike_is_actually_a_spike():
    """A flag described as a spike must report a ratio of at least three times its own normal."""
    anomalies = pd.read_csv(config.ANOMALIES_CSV)
    tagged = anomalies[anomalies["rule_tags"].fillna("").str.contains("Volume spike")]
    assert len(tagged) > 0
    for value in tagged["rule_tags"]:
        for tag in ast.literal_eval(value):
            if tag.startswith("Volume spike"):
                ratio = float(tag.split("volume ")[1].split("x")[0])
                assert ratio >= 3.0, f"spike reported below 3x: {tag}"


def test_rule_tags_render_readable_text():
    from train_anomaly import rule_tags

    row = pd.Series({"relight": True, "gap_years": 3, "volume_spike": True, "ratio_to_site_median": 3.4})
    tags = rule_tags(row)
    assert any("Relight" in tag and "3 dark years" in tag for tag in tags)
    assert any("Volume spike" in tag for tag in tags)


@pytest.mark.skipif(not config.PANEL_CSV.exists(), reason="pipeline output not built yet")
def test_built_panel_is_dense_and_zero_filled():
    panel = pd.read_csv(config.PANEL_CSV)
    expected = panel["id"].nunique() * (config.YEAR_END - config.YEAR_START + 1)
    assert len(panel) == expected
    assert panel["BCM"].min() >= 0


@pytest.mark.skipif(not config.CO2E_CSV.exists(), reason="pipeline output not built yet")
def test_co2e_file_is_not_stale_relative_to_panel():
    """The CO2e file is written from the panel, so its volumes must still match row for row."""
    panel = pd.read_csv(config.PANEL_CSV).sort_values(["id", "year"]).reset_index(drop=True)
    co2e = pd.read_csv(config.CO2E_CSV).sort_values(["id", "year"]).reset_index(drop=True)
    assert len(panel) == len(co2e)
    assert panel["BCM"].to_numpy() == pytest.approx(co2e["BCM"].to_numpy())
    for column in ("site_median_bcm", "ratio_to_site_median", "dark_run", "gap_years"):
        if column in panel.columns and column in co2e.columns:
            assert panel[column].fillna(-1).to_numpy() == pytest.approx(co2e[column].fillna(-1).to_numpy()), f"{column} is stale in {config.CO2E_CSV.name}"


@pytest.mark.skipif(not config.NATIONAL_JSON.exists(), reason="pipeline output not built yet")
def test_national_summary_has_headline_numbers():
    import json

    summary = json.loads(config.NATIONAL_JSON.read_text(encoding="utf-8"))
    assert summary["latest_year"]["bcm"] > 0
    assert summary["cumulative"]["co2e_mt"] > 0
    assert summary["emission_factor_t_co2e_per_m3"] == config.CO2E_T_PER_M3