"""Paths, source URLs and conversion constants for the Flare Watch Nigeria pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "models" / "artifacts"

EOG_BASE = "https://eogdata.mines.edu/global_flare_data/"

# The World Bank tracker is republished each year behind a new document hash.
# Update WB_DOC_ID and WB_YEARS when a newer edition of the tracker is published.
WB_DOC_ID = "b34e0c054bb3fe3695e70154c28eef3f-0400072026"
WB_YEARS = "2012-2025"
WB_SITES_URL = (
    f"https://thedocs.worldbank.org/en/doc/{WB_DOC_ID}"
    f"/related/Flare-Volume-Estimates-by-individual-Flare-Location-{WB_YEARS}.xlsx"
)
WB_COUNTRY_URL = (
    f"https://thedocs.worldbank.org/en/doc/{WB_DOC_ID}"
    f"/related/Flare-volume-and-intensity-estimates-{WB_YEARS}.xlsx"
)

# EOG annual flaring workbooks. The 2012-2016 workbook uses a different headerless
# layout and is intentionally excluded; the World Bank spine already covers those years.
EOG_FILES = {
    2017: "VIIRS_Global_flaring_d.7_slope_0.029353_2017_web_v1.xlsx",
    2018: "VIIRS_Global_flaring_d.7_slope_0.029353_2018_web.xlsx",
    2019: "VIIRS_Global_flaring_d.7_slope_0.029353_2019_web_v20201114.xlsx",
    2020: "VIIRS_Global_flaring_d.7_slope_0.029353_2020_web_v1.xlsx",
    2021: "VIIRS_Global_flaring_d.7_slope_0.029353_2021_web.xlsx",
    2022: "VIIRS_Global_flaring_d.7_slope_0.029353_2022_v20230526_web.xlsx",
    2023: "VIIRS_Global_flaring_d.7_slope_0.029353_2023_v20230614_web_IDmatch.xlsx",
    2024: "VIIRS_Global_flaring_d.7_slope_0.029353_2024_v20240730_web_IDmatch.xlsx",
}

BOUNDARIES_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/NGA/ADM1/"
    "geoBoundaries-NGA-ADM1.geojson"
)
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_BBOX = (3.3, 2.3, 14.5, 15.5)  # south, west, north, east

USER_AGENT = "flare-watch-nigeria/1.0 (satellite flaring anomaly tracker; contact via GitHub)"

COUNTRY = "Nigeria"
YEAR_START, YEAR_END = 2012, 2025

# Emission factor implied by the World Bank's own 2023 pairing of 148 bcm with 381 Mt CO2e,
# which works out to 2.57 kg CO2e per cubic metre of gas flared.
CO2E_KG_PER_M3 = 2.57
CO2E_T_PER_M3 = CO2E_KG_PER_M3 / 1000.0
KWH_THERMAL_PER_M3 = 10.55
PLANT_EFFICIENCY = 0.38
HOUSEHOLD_KWH_PER_YEAR = 250

SITES_CSV = PROCESSED / "sites.csv"
PANEL_CSV = PROCESSED / "site_year_panel.csv"
FEATURES_CSV = PROCESSED / "features.csv"
CO2E_CSV = PROCESSED / "site_year_co2e.csv"
NATIONAL_JSON = PROCESSED / "national_summary.json"
ANOMALIES_CSV = PROCESSED / "anomalies.csv"
SITE_RISK_CSV = PROCESSED / "site_risk.csv"
MODEL_PKL = ARTIFACTS / "isolation_forest.pkl"
FEATURE_STATS_JSON = ARTIFACTS / "feature_stats.json"