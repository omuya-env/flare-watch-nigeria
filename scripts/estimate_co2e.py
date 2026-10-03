"""Convert panel volumes to CO2e and compute the national headline figures."""

import argparse
import json

import pandas as pd

import config


def to_co2e_tonnes(bcm: float) -> float:
    """Convert billion cubic metres of flared gas to tonnes of CO2e."""
    return bcm * 1e9 * config.CO2E_T_PER_M3


def to_kwh_electricity(m3: float) -> float:
    """Convert cubic metres of flared gas to electricity it could have generated."""
    return m3 * config.KWH_THERMAL_PER_M3 * config.PLANT_EFFICIENCY


def homes_equivalent(m3: float) -> float:
    """Convert cubic metres of flared gas into the number of households it could power for a year."""
    return to_kwh_electricity(m3) / config.HOUSEHOLD_KWH_PER_YEAR


def read_country_totals(path) -> pd.DataFrame:
    """Read the World Bank country workbook, which stores one column per year, for the calibration check."""
    if not path.exists():
        return pd.DataFrame()
    book = pd.read_excel(path, sheet_name=None)
    for name, table in book.items():
        if "volume" not in str(name).lower():
            continue
        table = table.copy()
        table.columns = [str(column).strip() for column in table.columns]
        country_column = table.columns[0]
        match = table[table[country_column].astype(str).str.strip().eq(config.COUNTRY)]
        if match.empty:
            continue
        row = match.iloc[0]
        return pd.DataFrame({"year": [int(c) for c in table.columns[1:] if str(c).isdigit()],
                              "published_bcm": [float(row[c]) for c in table.columns[1:] if str(c).isdigit()]})
    return pd.DataFrame()


def cross_check(national: pd.DataFrame, published: pd.DataFrame) -> dict | None:
    """Compare pipeline national totals against the published country table."""
    if published.empty:
        return None
    merged = national[["year", "bcm"]].merge(published, on="year", how="inner")
    merged = merged[merged["published_bcm"] > 0]
    if merged.empty:
        return None
    merged["difference_pct"] = (merged["bcm"] - merged["published_bcm"]).abs() / merged["published_bcm"] * 100
    return {
        "source": "World Bank country workbook",
        "years_compared": int(len(merged)),
        "mean_difference_pct": float(merged["difference_pct"].mean()),
        "max_difference_pct": float(merged["difference_pct"].max()),
        "per_year": {str(int(r.year)): round(float(r.difference_pct), 3) for r in merged.itertuples()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country-file", default=None)
    args = parser.parse_args()

    panel = pd.read_csv(config.PANEL_CSV)
    panel["co2e_tonnes"] = panel["BCM"].map(to_co2e_tonnes)
    panel["m3_flared"] = panel["BCM"] * 1e9
    panel["kwh_equivalent"] = panel["m3_flared"].map(to_kwh_electricity)
    panel["homes_equivalent"] = panel["kwh_equivalent"] / config.HOUSEHOLD_KWH_PER_YEAR
    panel.to_csv(config.CO2E_CSV, index=False)

    national = panel.groupby("year").agg(
        bcm=("BCM", "sum"),
        m3_flared=("m3_flared", "sum"),
        co2e_tonnes=("co2e_tonnes", "sum"),
        kwh_equivalent=("kwh_equivalent", "sum"),
        active_sites=("active", "sum"),
    )
    national["homes_equivalent"] = national["kwh_equivalent"] / config.HOUSEHOLD_KWH_PER_YEAR
    national = national.reset_index()

    peak = national.loc[national["bcm"].idxmax()]
    latest = national.iloc[-1]
    summary = {
        "emission_factor_t_co2e_per_m3": config.CO2E_T_PER_M3,
        "kwh_thermal_per_m3": config.KWH_THERMAL_PER_M3,
        "plant_efficiency": config.PLANT_EFFICIENCY,
        "household_kwh_per_year": config.HOUSEHOLD_KWH_PER_YEAR,
        "coverage": {"first_year": int(national["year"].min()), "last_year": int(national["year"].max())},
        "peak_year": {"year": int(peak["year"]), "bcm": float(peak["bcm"]), "co2e_mt": float(peak["co2e_tonnes"] / 1e6)},
        "latest_year": {
            "year": int(latest["year"]),
            "bcm": float(latest["bcm"]),
            "co2e_mt": float(latest["co2e_tonnes"] / 1e6),
            "active_sites": int(latest["active_sites"]),
            "homes_equivalent": float(latest["homes_equivalent"]),
        },
        "cumulative": {
            "bcm": float(national["bcm"].sum()),
            "co2e_mt": float(national["co2e_tonnes"].sum() / 1e6),
            "homes_equivalent": float(national["homes_equivalent"].sum()),
        },
    }

    country_file = args.country_file or (config.RAW / "worldbank_flare_country.xlsx")
    check = cross_check(national, read_country_totals(country_file))
    if check is not None:
        summary["cross_check"] = check

    config.NATIONAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.NATIONAL_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    national.to_csv(config.PROCESSED / "national_yearly.csv", index=False)

    print(national.assign(bcm=national["bcm"].round(2), co2e_mt=(national["co2e_tonnes"] / 1e6).round(2), homes=(national["homes_equivalent"] / 1e6).round(1)).to_string(index=False))
    print(f"\npeak year {int(peak['year'])}: {peak['bcm']:.2f} bcm, {peak['co2e_tonnes'] / 1e6:.1f} Mt CO2e")
    print(f"cumulative {national['bcm'].sum():.1f} bcm = {national['co2e_tonnes'].sum() / 1e6:.0f} Mt CO2e")
    if "cross_check" in summary:
        check_result = summary["cross_check"]
        print(f"cross-check vs World Bank country file: {check_result['years_compared']} years, mean diff {check_result['mean_difference_pct']:.3f}%, max diff {check_result['max_difference_pct']:.3f}%")
    print(f"wrote {config.CO2E_CSV.relative_to(config.ROOT)} and {config.NATIONAL_JSON.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()