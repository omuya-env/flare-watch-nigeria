"""Build the Nigeria flare site registry from the World Bank per-location workbook."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from shapely.geometry import Point, shape

import config

NA_TOKENS = {"NA", "N/A", "", "None", "nan", "-", "null"}


def clean_series(series: pd.Series) -> pd.Series:
    """Replace the string placeholders the workbook uses for missing metadata."""
    return series.where(~series.astype(str).str.strip().isin(NA_TOKENS), other=pd.NA)


def read_nigeria_sites(path: Path) -> pd.DataFrame:
    """Stream the workbook and keep only Nigerian rows so the global table never lands in memory."""
    workbook = load_workbook(path, read_only=True, data_only=True)
    wanted = ("id", "lat", "lon", "Year", "BCM", "MMSCFD", "Field type", "Operator", "Field name", "Location", "Flare Level")
    sheet = None
    header: list[str] = []
    for name in workbook.sheetnames:
        candidate = workbook[name]
        first_row = next(candidate.iter_rows(values_only=True), None)
        if first_row is None:
            continue
        labels = [str(value).strip() if value else "" for value in first_row]
        if "Country" in labels and all(column in labels for column in wanted):
            sheet, header = candidate, labels
            break
    if sheet is None:
        workbook.close()
        raise ValueError(f"no sheet in {path.name} has the expected flare-location columns")
    rows = sheet.iter_rows(values_only=True)
    next(rows)
    country_index = header.index("Country")
    keep = [header.index(name) for name in wanted]
    collected = []
    for row in rows:
        if len(row) <= country_index or row[country_index] != config.COUNTRY:
            continue
        collected.append(tuple(row[i] if i < len(row) else None for i in keep))
    workbook.close()
    frame = pd.DataFrame(collected, columns=[header[i] for i in keep])
    for column in ("Operator", "Field name", "Field type", "Location", "Flare Level"):
        frame[column] = clean_series(frame[column])
    for column in ("lat", "lon", "Year", "BCM", "MMSCFD"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["id", "lat", "lon", "Year"]).reset_index(drop=True)
    frame["id"] = frame["id"].astype(str)
    frame["Year"] = frame["Year"].astype(int)
    return frame


def load_states(path: Path):
    """Return a list of (state name, polygon) pairs from the geoBoundaries ADM1 file."""
    if not path.exists():
        print("  ! boundaries file missing, state will be blank (run scripts/download_data.py)")
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    states = []
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        name = properties.get("shapeName") or properties.get("shapeISO") or properties.get("Name")
        if not name:
            continue
        geometry = feature.get("geometry")
        if geometry:
            states.append((str(name), shape(geometry)))
    return states


def assign_states(frame: pd.DataFrame, states: list) -> pd.Series:
    """Label each site with the state polygon that contains it."""
    labels = []
    for lat, lon in zip(frame["lat"], frame["lon"]):
        point = Point(lon, lat)
        found = ""
        for name, polygon in states:
            if polygon.covers(point):
                found = name
                break
        labels.append(found)
    return pd.Series(labels, index=frame.index)


def load_settlements(path: Path) -> pd.DataFrame:
    if not path.exists():
        print("  ! settlement file missing, nearest community will be blank")
        return pd.DataFrame(columns=["name", "lat", "lon", "place"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for element in payload.get("elements", []):
        tags = element.get("tags") or {}
        name = tags.get("name")
        if name and "lat" in element:
            records.append({"name": name, "lat": element["lat"], "lon": element["lon"], "place": tags.get("place", "")})
    return pd.DataFrame(records)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def nearest_settlement(lat, lon, places: pd.DataFrame) -> tuple[str, float]:
    if places.empty:
        return "", float("nan")
    distances = [haversine_km(lat, lon, row.lat, row.lon) for row in places.itertuples()]
    best = int(min(range(len(distances)), key=distances.__getitem__))
    return places.iloc[best]["name"], distances[best]


def build_registry(observations: pd.DataFrame, states: list, places: pd.DataFrame) -> pd.DataFrame:
    """Collapse the site-year observations into one row per flare site."""
    grouped = observations.groupby("id")
    registry = grouped.agg(
        lat=("lat", "median"),
        lon=("lon", "median"),
        first_year=("Year", "min"),
        last_year=("Year", "max"),
        active_years=("Year", "nunique"),
        total_bcm=("BCM", "sum"),
        peak_bcm=("BCM", "max"),
    ).reset_index()
    peak_year = grouped.apply(lambda g: int(g.loc[g["BCM"].idxmax(), "Year"]), include_groups=False)
    registry["peak_year"] = registry["id"].map(peak_year)
    for column in ("Field type", "Operator", "Field name", "Location", "Flare Level"):
        registry[column] = registry["id"].map(grouped[column].first())
    registry = registry.rename(
        columns={"Field type": "field_type", "Operator": "operator", "Field name": "field_name", "Location": "location", "Flare Level": "flare_level"}
    )
    registry["state"] = assign_states(registry, states)
    offshore = registry["location"].astype(str).str.upper().eq("OFFSHORE")
    registry["state"] = registry["state"].where(registry["state"].ne(""), pd.NA)
    registry["state"] = np.where(offshore & registry["state"].isna(), "Offshore", registry["state"])
    nearest = [nearest_settlement(row.lat, row.lon, places) for row in registry.itertuples()]
    registry["nearest_settlement"] = [n for n, _ in nearest]
    registry["nearest_settlement_km"] = [d for _, d in nearest]
    registry["active_span_years"] = registry["last_year"] - registry["first_year"] + 1
    registry["mean_bcm_per_active_year"] = registry["total_bcm"] / registry["active_years"]
    return registry.sort_values("total_bcm", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=config.RAW / "worldbank_flare_sites.xlsx")
    args = parser.parse_args()

    print(f"reading Nigeria slice from {args.raw.name}")
    observations = read_nigeria_sites(args.raw)
    print(f"  site-year rows: {len(observations):,} across {observations['id'].nunique():,} sites, {observations['Year'].min()} to {observations['Year'].max()}")
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    long_slice = config.PROCESSED / "site_year_raw.csv"
    observations.to_csv(long_slice, index=False)
    print(f"  wrote {long_slice.relative_to(config.ROOT)}")

    states = load_states(config.RAW / "nga_adm1.geojson")
    places = load_settlements(config.RAW / "osm_settlements.json")
    print(f"  state polygons: {len(states)} | named places: {len(places):,}")

    registry = build_registry(observations, states, places)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    registry.to_csv(config.SITES_CSV, index=False)

    unassigned = int((registry["state"] == "").sum())
    print(f"\nwrote {config.SITES_CSV.relative_to(config.ROOT)}  rows={len(registry)}  unassigned_state={unassigned}")
    print(registry.head(5)[["id", "state", "operator", "field_name", "total_bcm", "active_years"]].to_string(index=False))


if __name__ == "__main__":
    main()