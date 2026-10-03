"""Generate the pipeline walkthrough notebook so it can be regenerated rather than hand-edited."""

from pathlib import Path

import nbformat as nbf

CELLS = [
    nbf.v4.new_markdown_cell(
        "# Flare Watch Nigeria: pipeline walkthrough\n"
        "\n"
        "This notebook shows how the processed data was built, step by step. It reads the files that\n"
        "`scripts/` and `models/train_anomaly.py` produced, so run the pipeline first:\n"
        "\n"
        "```\n"
        "python scripts/download_data.py\n"
        "python scripts/build_sites.py\n"
        "python scripts/build_timeseries.py\n"
        "python scripts/estimate_co2e.py\n"
        "python models/train_anomaly.py\n"
        "```\n"
        "\n"
        "Every source file is public. The raw downloads are cached in `data/raw/` and are not committed."
    ),
    nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import sys\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT / 'scripts'))\n"
        "PROCESSED = ROOT / 'data' / 'processed'\n"
        "pd.set_option('display.width', 160)\n"
        "pd.set_option('display.max_columns', 30)\n"
        "print('project root:', ROOT)"
    ),
    nbf.v4.new_markdown_cell(
        "## 1. The site registry\n"
        "\n"
        "`scripts/build_sites.py` streams the World Bank per-location workbook, keeps only Nigerian rows,\n"
        "assigns each site to a state using point-in-polygon against geoBoundaries ADM1 polygons, and finds\n"
        "the nearest named settlement from OpenStreetMap."
    ),
    nbf.v4.new_code_cell(
        "sites = pd.read_csv(PROCESSED / 'sites.csv')\n"
        "print(f'{len(sites):,} flare sites, {sites[\"first_year\"].min()} to {sites[\"last_year\"].max()}')\n"
        "print()\n"
        "sites.groupby('state', dropna=False)['total_bcm'].sum().sort_values(ascending=False).head(10)"
    ),
    nbf.v4.new_code_cell(
        "sites.nlargest(12, 'total_bcm')[['id', 'state', 'operator', 'field_name', 'first_year', 'last_year', 'total_bcm', 'peak_bcm', 'nearest_settlement', 'nearest_settlement_km']]"
    ),
    nbf.v4.new_markdown_cell(
        "Operator and field name come straight from the World Bank metadata. They are shown for context and\n"
        "are never used to group, match or attribute anything."
    ),
    nbf.v4.new_markdown_cell(
        "## 2. The annual panel\n"
        "\n"
        "`scripts/build_timeseries.py` expands every site across every year and derives the continuity,\n"
        "persistence and comparison features the anomaly model consumes."
    ),
    nbf.v4.new_code_cell(
        "panel = pd.read_csv(PROCESSED / 'site_year_panel.csv')\n"
        "print(f'{len(panel):,} site-years = {panel[\"id\"].nunique():,} sites x {panel[\"year\"].nunique()} years')\n"
        "print(f'active {panel[\"active\"].sum():,}   dark {len(panel) - panel[\"active\"].sum():,}')\n"
        "print(f'relights {panel[\"relight\"].sum():,}   shutdowns {panel[\"shutdown\"].sum():,}   spikes {panel[\"volume_spike\"].sum():,}')\n"
        "panel.head()"
    ),
    nbf.v4.new_code_cell(
        "figure, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "yearly = panel.groupby('year')['BCM'].sum()\n"
        "axes[0].bar(yearly.index, yearly.values, color='#2563eb')\n"
        "axes[0].set_title('Nigeria flared gas by year')\n"
        "axes[0].set_xlabel('year')\n"
        "axes[0].set_ylabel('billion cubic metres')\n"
        "axes[1].hist(panel.loc[panel['detection_freq'].notna(), 'detection_freq'], bins=30, color='#ef4444')\n"
        "axes[1].set_title('Detection frequency (share of clear-sky passes)')\n"
        "axes[1].set_xlabel('detection frequency')\n"
        "plt.tight_layout()"
    ),
    nbf.v4.new_markdown_cell(
        "## 3. Carbon and energy\n"
        "\n"
        "The emission factor is 2.57 kg CO2e per cubic metre of gas flared, which is what the World Bank's\n"
        "own global figures imply: 148 bcm paired with 381 Mt CO2e in 2023."
    ),
    nbf.v4.new_code_cell(
        "co2e = pd.read_csv(PROCESSED / 'site_year_co2e.csv')\n"
        "summary = pd.read_json(PROCESSED / 'national_summary.json', typ='series')\n"
        "national = co2e.groupby('year').agg(bcm=('BCM', 'sum'), co2e_mt=('co2e_tonnes', lambda s: s.sum() / 1e6), homes_m=('homes_equivalent', lambda s: s.sum() / 1e6))\n"
        "national.round(2)"
    ),
    nbf.v4.new_markdown_cell(
        "## 4. Anomalies\n"
        "\n"
        "A single Isolation Forest is fitted on every flaring site-year, and five deterministic rules catch\n"
        "the behaviours regulators care about: relights, volume spikes, extended shutdowns, persistent flares\n"
        "and new sites."
    ),
    nbf.v4.new_code_cell(
        "anomalies = pd.read_csv(PROCESSED / 'anomalies.csv')\n"
        "print(f'{len(anomalies):,} flagged site-years across {anomalies[\"id\"].nunique():,} sites')\n"
        "anomalies['flag_source'].value_counts()"
    ),
    nbf.v4.new_code_cell(
        "sample = anomalies[anomalies['rule_tags'].notna()].iloc[0]\n"
        "print(f'site {sample[\"id\"]} in {int(sample[\"year\"])}, {sample[\"state\"]}')\n"
        "print('rules :', sample['rule_tags'])\n"
        "print('model :', sample['shap_explanation'])"
    ),
    nbf.v4.new_markdown_cell("## 5. The worst flagged sites\n"),
    nbf.v4.new_code_cell(
        "risk = pd.read_csv(PROCESSED / 'site_risk.csv')\n"
        "risk.head(15)[['id', 'state', 'operator', 'field_name', 'flags', 'worst_score', 'latest_flag_year', 'total_bcm', 'nearest_settlement', 'nearest_settlement_km']]"
    ),
    nbf.v4.new_code_cell(
        "top = risk.nlargest(20, 'total_bcm')\n"
        "figure, axes = plt.subplots(1, 2, figsize=(13, 5))\n"
        "scatter = axes[0].scatter(top['lon'], top['lat'], s=top['total_bcm'] * 90, c=top['worst_score'], cmap='YlOrRd', alpha=0.75, edgecolor='white')\n"
        "figure.colorbar(scatter, ax=axes[0], label='highest anomaly score')\n"
        "axes[0].set_title('20 largest flare sites, shaded by anomaly score')\n"
        "axes[0].set_xlabel('longitude')\n"
        "axes[0].set_ylabel('latitude')\n"
        "counts = anomalies['rule_tags'].dropna().str.count(',') + 1\n"
        "axes[1].hist(counts, bins=range(1, counts.max() + 2), color='#2563eb', align='left')\n"
        "axes[1].set_title('How many rules fired per flagged year')\n"
        "axes[1].set_xlabel('rules fired')\n"
        "axes[1].set_ylabel('flagged site-years')\n"
        "plt.tight_layout()"
    ),
    nbf.v4.new_markdown_cell(
        "## What this does not claim\n"
        "\n"
        "Flare volumes are satellite-derived estimates calibrated against country-level reporting, not meter\n"
        "readings. A year with no detection means VIIRS saw no flare, which usually means no flaring, but can\n"
        "also mean cloud cover or a plume below the detection threshold. Absence of evidence is not evidence of\n"
        "absence, and a shutdown flag is a signal to check, not a finding of compliance."
    ),
]

notebook = nbf.v4.new_notebook(cells=CELLS)
notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
notebook.metadata["kernelspec"]["name"] = "python3"

target = Path(__file__).resolve().parent.parent / "notebooks" / "01_pipeline_walkthrough.ipynb"
target.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, str(target))
print(f"wrote {target}")