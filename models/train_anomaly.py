"""Fit the Isolation Forest, apply the rule based flags and write plain-language SHAP explanations."""

import argparse
import json
import pickle

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import config  # noqa: E402

MODEL_FEATURES = [
    "BCM",
    "bcm_yoy_change",
    "ratio_to_site_median",
    "detection_freq",
    "avg_temp_k",
    "gap_years",
    "volume_zscore",
    "persistence_zscore",
]

FEATURE_LABELS = {
    "BCM": "flare volume",
    "bcm_yoy_change": "year on year volume change",
    "ratio_to_site_median": "volume relative to its own median",
    "detection_freq": "detection frequency",
    "avg_temp_k": "flame temperature",
    "gap_years": "years since the site last flared",
    "volume_zscore": "volume against other sites this year",
    "persistence_zscore": "persistence against other sites",
}

RULES = {
    "relight": ("Relight", "flaring restarted after {gap_years} dark year(s)"),
    "volume_spike": ("Volume spike", "volume {ratio:.1f}x its own recent normal"),  # ratio is against the trailing three flaring years
    "extended_shutdown": ("Extended shutdown", "{dark_run} consecutive dark years after flaring"),
    "high_persistence": ("Persistent flare", "detected in {detection_freq:.0%} of clear-sky passes"),
    "new_site": ("New site", "first flare ever seen at this site"),
}


def prepare_matrix(frame: pd.DataFrame, reference: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Median-impute the nullable remote-sensing features, taking fill values from flaring rows only."""
    reference = frame if reference is None else reference
    stats = {"medians": {}, "missing_counts": {}}
    for column in MODEL_FEATURES:
        stats["medians"][column] = float(reference[column].median()) if reference[column].notna().any() else 0.0
        stats["missing_counts"][column] = int(frame[column].isna().sum())
    matrix = frame[MODEL_FEATURES].fillna(pd.Series(stats["medians"]))
    return matrix.astype(float), stats


def describe_feature(index: int, value: float, typical: float) -> str:
    """Say what one feature actually was, in the units a reader cares about."""
    column = MODEL_FEATURES[index]
    if column == "BCM":
        return f"flare volume {value:.4f} bcm against {typical:.4f} typical"
    if column == "bcm_yoy_change":
        return f"volume {'rose' if value >= 0 else 'fell'} {abs(value):.0%} year on year"
    if column == "ratio_to_site_median":
        return f"volume {value:.2f}x its own median"
    if column == "detection_freq":
        return f"seen in {value:.0%} of clear-sky passes against {typical:.0%} typical"
    if column == "avg_temp_k":
        return f"flame temperature {value:.0f} K against {typical:.0f} K typical"
    if column == "gap_years":
        return f"{value:.0f} years since this site last flared"
    if column == "volume_zscore":
        return f"volume {abs(value):.1f} standard deviations {'below' if value < 0 else 'above'} other sites that year"
    if column == "persistence_zscore":
        return f"persistence {abs(value):.1f} standard deviations {'below' if value < 0 else 'above'} other sites"
    return f"{FEATURE_LABELS[column]} {value:.3g} against {typical:.3g} typical"


def explain_shap(model, matrix: pd.DataFrame, typical: dict, top_k: int = 3) -> list[str]:
    """Turn the top SHAP contributions of each row into one readable sentence."""
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(matrix)
    if isinstance(values, list):
        values = values[0]
    numbers = matrix.to_numpy()
    sentences = []
    for row_index in range(len(matrix)):
        row_values = values[row_index]
        order = np.argsort(np.abs(row_values))[::-1][:top_k]
        parts = [describe_feature(int(i), float(numbers[row_index][i]), float(typical[MODEL_FEATURES[i]])) for i in order]
        sentences.append("top drivers: " + "; ".join(parts))
    return sentences


def rule_tags(row: pd.Series) -> list[str]:
    """Collect the deterministic flags that fired for one site-year."""
    tags = []
    for key, (_, template) in RULES.items():
        if bool(row.get(key, False)):
            if key == "relight":
                years = int(row["gap_years"])
                tags.append(f"Relight: flaring restarted after {years} dark year{'s' if years != 1 else ''}")
            elif key == "volume_spike":
                trailing = float(row.get("trailing_median_bcm", float("nan")))
                ratio = float(row["BCM"]) / trailing if trailing > 0 else float("nan")
                tags.append(f"Volume spike: volume {ratio:.1f}x its own recent normal")
            elif key == "extended_shutdown":
                tags.append(f"Extended shutdown: {template.format(dark_run=int(row['dark_run']))}")
            elif key == "high_persistence":
                tags.append(f"Persistent flare: {template.format(detection_freq=float(row['detection_freq']))}")
            else:
                tags.append(f"New site: {template}")
    return tags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contamination", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    panel = pd.read_csv(config.PANEL_CSV)
    sites = pd.read_csv(config.SITES_CSV)
    flaring = panel[panel["active"]]
    matrix, stats = prepare_matrix(panel, reference=flaring)
    fit_mask = panel["active"].to_numpy()

    model = IsolationForest(n_estimators=300, contamination=args.contamination, random_state=args.seed, n_jobs=-1)
    model.fit(matrix.to_numpy()[fit_mask])
    raw_score = model.decision_function(matrix.to_numpy())
    panel["anomaly_score"] = np.where(fit_mask, (raw_score.max() - raw_score) / (raw_score.max() - raw_score.min()), 0.0)

    shap_sentences = explain_shap(model, matrix, stats["medians"])
    panel["shap_explanation"] = shap_sentences
    panel["rule_tags"] = panel.apply(rule_tags, axis=1)
    model_flag = fit_mask & (raw_score < 0)
    rule_flag = panel["rule_tags"].str.len().gt(0)
    panel["flagged"] = model_flag | rule_flag
    panel["flag_source"] = np.where(
        rule_flag & model_flag,
        "model and rule",
        np.where(rule_flag, "rule", np.where(model_flag, "model", "")),
    )

    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(config.MODEL_PKL, "wb") as handle:
        pickle.dump({"model": model, "features": MODEL_FEATURES, "medians": stats["medians"]}, handle)
    config.FEATURE_STATS_JSON.write_text(json.dumps({"features": MODEL_FEATURES, **stats}, indent=2), encoding="utf-8")

    flagged = panel[panel["flagged"]].copy()
    columns = [
        "id", "year", "state", "operator", "field_name", "BCM", "detection_freq", "avg_temp_k",
        "gap_years", "dark_run", "anomaly_score", "flag_source", "rule_tags", "shap_explanation",
    ]
    flagged[columns].sort_values("anomaly_score", ascending=False).to_csv(config.ANOMALIES_CSV, index=False)

    risk = (
        flagged.groupby("id")
        .agg(flags=("year", "count"), worst_score=("anomaly_score", "max"), latest_flag_year=("year", "max"))
        .reset_index()
        .merge(sites[["id", "lat", "lon", "state", "operator", "field_name", "flare_level", "nearest_settlement", "nearest_settlement_km", "total_bcm"]], on="id", how="left")
        .sort_values("worst_score", ascending=False)
    )
    risk.to_csv(config.SITE_RISK_CSV, index=False)

    print(f"model fitted on {int(fit_mask.sum()):,} flaring site-years with {len(MODEL_FEATURES)} features")
    print(f"model outliers (score<0): {int(model_flag.sum()):,}   rule hits: {int(rule_flag.sum()):,}")
    print(f"total flagged rows: {len(flagged):,}   sites flagged: {flagged['id'].nunique():,}")
    counts = flagged["flag_source"].value_counts().to_dict()
    print(f"flag sources: {counts}")
    print(f"wrote {config.ANOMALIES_CSV.relative_to(config.ROOT)}, {config.SITE_RISK_CSV.relative_to(config.ROOT)}, {config.MODEL_PKL.relative_to(config.ROOT)}")
    if len(flagged):
        print("\ntop examples:")
        for row in flagged.nlargest(3, "anomaly_score").itertuples():
            print(f"  site {row.id} {row.year} score={row.anomaly_score:.2f} {row.state or ''} {row.shap_explanation}")


if __name__ == "__main__":
    main()