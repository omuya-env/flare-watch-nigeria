"""Build the dense site-year panel and the engineered features the anomaly model consumes."""

import argparse

import numpy as np
import pandas as pd
from openpyxl import load_workbook

import config

EOG_COLUMNS = {
    "lat": ("latitude",),
    "lon": ("longitude",),
    "bcm": ("bcm",),
    "avg_temp_k": ("avg_temp",),
    "ellipticity": ("ellipticity",),
    "detection_freq": ("detection_freq", "detectionfrequency"),
    "clear_obs": ("clear_obs", "clearobs"),
}


def normalise_header(name) -> str:
    """Reduce an EOG header cell to a comparable token, since naming drifts year to year."""
    text = str(name).strip().lower().replace(".", "").replace(",", "").replace(" ", "_").replace("-", "_")
    return text.strip("_")


def resolve_columns(header: list[str]) -> dict:
    """Map each logical field to a column index by prefix match."""
    positions = {}
    for field, prefixes in EOG_COLUMNS.items():
        for index, name in enumerate(header):
            if any(name.startswith(prefix) for prefix in prefixes):
                positions[field] = index
                break
    return positions


def read_eog_year(path, year: int) -> pd.DataFrame:
    """Pull Nigeria rows for one year out of an EOG annual flaring workbook."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    frames = []
    for sheet_name in workbook.sheetnames:
        rows = workbook[sheet_name].iter_rows(values_only=True)
        first = next(rows, None)
        if first is None:
            continue
        header = [normalise_header(name) for name in first]
        if "country" not in header or "latitude" not in header or "longitude" not in header:
            continue
        country_index = header.index("country")
        positions = resolve_columns(header)
        collected = []
        for row in rows:
            if len(row) <= country_index or row[country_index] != config.COUNTRY:
                continue
            record = {field: (row[index] if index < len(row) else None) for field, index in positions.items()}
            record["year"] = year
            collected.append(record)
        if collected:
            frames.append(pd.DataFrame(collected))
    workbook.close()
    if not frames:
        return pd.DataFrame(columns=list(EOG_COLUMNS) + ["year"])
    combined = pd.concat(frames, ignore_index=True)
    for column in EOG_COLUMNS:
        combined[column] = pd.to_numeric(combined.get(column), errors="coerce")
    if combined["detection_freq"].notna().any() and combined["detection_freq"].median() > 1.5:
        combined["detection_freq"] = combined["detection_freq"] / 100.0
    return combined


def match_eog_to_sites(eog: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Attach each EOG detection to the nearest registry site, within one grid cell."""
    columns = ["id", "year", "avg_temp_k", "ellipticity", "detection_freq", "clear_obs"]
    if eog.empty:
        return pd.DataFrame(columns=columns)
    site_ids = sites["id"].to_numpy()
    site_lat = sites["lat"].to_numpy("float64")
    site_lon = sites["lon"].to_numpy("float64")
    records = []
    for row in eog.dropna(subset=["lat", "lon"]).itertuples():
        dy = np.radians(site_lat - row.lat)
        dx = np.radians(site_lon - row.lon) * np.cos(np.radians((site_lat + row.lat) / 2))
        distances = np.hypot(dx, dy)
        best = int(np.argmin(distances))
        if distances[best] <= 0.01:  # about 1.1 km, inside the 750 m flare grid spacing
            records.append(
                {
                    "id": site_ids[best],
                    "year": row.year,
                    "avg_temp_k": row.avg_temp_k,
                    "ellipticity": row.ellipticity if hasattr(row, "ellipticity") else np.nan,
                    "detection_freq": row.detection_freq,
                    "clear_obs": row.clear_obs if hasattr(row, "clear_obs") else np.nan,
                }
            )
    return pd.DataFrame(records, columns=columns)


def build_panel(long_slice: pd.DataFrame, sites: pd.DataFrame, eog: pd.DataFrame) -> pd.DataFrame:
    """Expand the observed site-years into a complete grid so dark years are explicit zeros."""
    years = pd.Index(range(config.YEAR_START, config.YEAR_END + 1), name="year")
    panel = pd.MultiIndex.from_product([sites["id"].to_numpy(), years], names=["id", "year"]).to_frame(index=False)
    active = long_slice[["id", "Year", "BCM", "MMSCFD"]].rename(columns={"Year": "year"})
    panel = panel.merge(active, on=["id", "year"], how="left")
    panel["active"] = panel["BCM"].fillna(0.0) > 0
    panel["BCM"] = panel["BCM"].fillna(0.0)
    if "MMSCFD" in panel:
        panel["MMSCFD"] = panel["MMSCFD"].fillna(0.0)
    panel = panel.merge(sites[["id", "state", "operator", "field_name", "flare_level", "field_type"]], on="id", how="left")
    if not eog.empty:
        eog_clean = eog.groupby(["id", "year"], as_index=False).agg(
            avg_temp_k=("avg_temp_k", "mean"),
            detection_freq=("detection_freq", "max"),
            ellipticity=("ellipticity", "mean"),
            clear_obs=("clear_obs", "max"),
        )
        panel = panel.merge(eog_clean, on=["id", "year"], how="left")
    panel = panel.sort_values(["id", "year"]).reset_index(drop=True)
    return add_features(panel)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Derive the continuity, persistence and cross-site comparison features."""
    panel = panel.sort_values(["id", "year"]).copy()
    grouped = panel.groupby("id", sort=False)

    run = grouped["active"].transform(lambda s: s.ne(s.shift()).cumsum())
    position = panel.groupby(["id", run]).cumcount()
    inactive = panel["active"].eq(False)
    panel["prev_active"] = grouped["active"].shift(1).fillna(False)
    previous_inactive = panel["prev_active"].eq(False)
    panel["dark_run"] = np.where(inactive, position + 1, 0)
    panel["gap_years"] = grouped["dark_run"].shift(1).fillna(0)
    panel["years_active"] = grouped["active"].cumsum()

    panel["relight"] = panel["active"] & previous_inactive & (panel["gap_years"] > 0)
    panel["new_site"] = panel["active"] & (panel["years_active"] == 1) & (panel["year"] > config.YEAR_START)
    panel["shutdown"] = inactive & panel["prev_active"]
    panel["extended_shutdown"] = inactive & previous_inactive & (panel["dark_run"] == 2)

    flaring_only = panel["BCM"].where(panel["active"])
    panel["site_median_bcm"] = flaring_only.groupby(panel["id"]).transform("median")
    panel["ratio_to_site_median"] = panel["BCM"] / panel["site_median_bcm"].replace(0, np.nan)
    previous = grouped["BCM"].shift(1)
    panel["bcm_yoy_change"] = (panel["BCM"] - previous) / previous.where(previous > 0)
    flaring_history = panel["BCM"].where(panel["active"])
    trailing = flaring_history.groupby(panel["id"]).transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
    panel["trailing_median_bcm"] = trailing
    panel["volume_spike"] = panel["active"] & (trailing > 0) & (panel["BCM"] >= 3.0 * trailing)

    year_mean = panel.groupby("year")["BCM"].transform("mean")
    year_std = panel.groupby("year")["BCM"].transform("std")
    panel["volume_zscore"] = (panel["BCM"] - year_mean) / year_std.where(year_std > 0)

    if "detection_freq" in panel.columns:
        spread = panel["detection_freq"].std()
        panel["persistence_zscore"] = (panel["detection_freq"] - panel["detection_freq"].mean()) / spread if spread else 0.0
        panel["high_persistence"] = panel["detection_freq"] >= panel["detection_freq"].quantile(0.95)
        panel["has_eog"] = panel["detection_freq"].notna()
    else:
        panel["persistence_zscore"] = 0.0
        panel["high_persistence"] = False
        panel["has_eog"] = False
    return panel


FEATURE_COLUMNS = [
    "id", "year", "BCM", "active", "detection_freq", "avg_temp_k",
    "gap_years", "relight", "shutdown", "volume_spike", "bcm_yoy_change",
    "ratio_to_site_median", "persistence_zscore", "volume_zscore",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    long_slice = pd.read_csv(config.PROCESSED / "site_year_raw.csv")
    sites = pd.read_csv(config.SITES_CSV)
    print(f"sites: {len(sites):,}   observed site-years: {len(long_slice):,}")

    eog_frames = []
    for year in sorted(config.EOG_FILES):
        path = config.RAW / "eog" / f"{year}.xlsx"
        if not path.exists():
            print(f"  {year}: workbook missing, year skipped (run scripts/download_data.py)")
            continue
        frame = read_eog_year(path, year)
        matched = match_eog_to_sites(frame, sites)
        print(f"  {year}: {len(frame):,} Nigeria detections -> {len(matched):,} matched to a site")
        eog_frames.append(matched)
    eog = pd.concat(eog_frames, ignore_index=True) if eog_frames else pd.DataFrame()

    panel = build_panel(long_slice, sites, eog)
    panel.to_csv(config.PANEL_CSV, index=False)
    panel[FEATURE_COLUMNS].to_csv(config.FEATURES_CSV, index=False)

    print(f"\npanel: {len(panel):,} rows = {panel['id'].nunique():,} sites x {panel['year'].nunique()} years")
    print(f"active site-years: {int(panel['active'].sum()):,}   relights: {int(panel['relight'].sum()):,}   shutdowns: {int(panel['shutdown'].sum()):,}")
    print(f"EOG-matched site-years: {int(panel['has_eog'].sum()):,}")
    print(f"wrote {config.PANEL_CSV.relative_to(config.ROOT)} and {config.FEATURES_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()