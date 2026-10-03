# Flare Watch Nigeria

Nigeria flared **6.60 billion cubic metres of gas in 2025**, about **17 million tonnes of CO2 equivalent**. Burned at a typical gas plant, that is roughly the electricity **106 million homes** use in a year. Nigeria has about 30 million households.

Nobody publishes that as a per-site story. The World Bank publishes national totals. NOSDRA publishes its own tracker. What is missing is the view a regulator, an operator or a community actually asks for: which flare sites are burning, which ones relit after they were shut down, which ones are suddenly running three times hotter than usual, and who lives nearest to them.

This project builds that view from public satellite data. No API keys, no subscriptions, no manual data entry.

## What it does

- **Maps every flare site** Nigeria has recorded, 476 of them, from 2012 to 2025, with annual volumes and state assignment.
- **Flags anomalies** with a single Isolation Forest plus five deterministic rules that regulators recognise: relights, volume spikes, extended shutdowns, persistent flares and new sites.
- **Explains every flag** in plain language using SHAP, for example *"volume 12.3 standard deviations above other sites that year; flare volume 0.7034 bcm against 0.0033 typical"*.
- **Converts volumes to CO2e and to energy**, using the emission factor the World Bank's own published figures imply.

## Quickstart

```bash
cd flare-watch-nigeria
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/build_sites.py
.venv/bin/python scripts/build_timeseries.py
.venv/bin/python scripts/estimate_co2e.py
.venv/bin/python models/train_anomaly.py

.venv/bin/streamlit run app/Home.py
```

The first run downloads about 66 MB of public data into `data/raw/`. Later runs reuse the cache. `build_sites.py` takes roughly two minutes because it streams a 47 MB global workbook; everything else is seconds.

## The three pages

| Page | What it answers |
|------|-----------------|
| **Home** | How much is Nigeria flaring, where, and which sites carry a flag |
| **Explorer** | Filter sites by state, operator, flare level, size and anomaly type, then export the result |
| **Site detail** | One site's full history, every flagged year with its explanation, and its carbon contribution |

Site detail pages are linkable: `/SiteDetail?site=s_3987`.

## How it works

```
data/raw/          public downloads, gitignored
  scripts/download_data.py     fetch and cache every source
  scripts/build_sites.py       Nigeria slice, states, nearest settlements
  scripts/build_timeseries.py  dense site-year panel and features
  scripts/estimate_co2e.py     CO2e, energy and headline numbers
models/train_anomaly.py        Isolation Forest, SHAP, rules
  app/                        Streamlit pages
data/processed/    Nigeria-only CSVs, committed so results are reproducible
notebooks/         walkthrough of the whole pipeline
tests/             unit tests for the continuity rules and conversions
```

## Data sources

| Source | Licence | What it provides |
|--------|---------|------------------|
| [World Bank Global Gas Flaring Tracker](https://www.worldbank.org/en/programs/gasflaringreduction/global-flaring-data) | Open | Per-site locations, annual volumes, operator and field metadata |
| [EOG VIIRS flaring workbooks](https://eogdata.mines.edu/products/vnf/global_gas_flare.html) | Free with attribution | Detection frequency and flame temperature, 2017 to 2024 |
| [geoBoundaries ADM1](https://www.geoboundaries.org/) | CC BY 4.0 | Nigeria state polygons |
| [OpenStreetMap via Overpass](https://overpass-api.de/) | ODbL | Nearest named settlement to each site |

Flare volumes originate from NOAA VIIRS satellite observations processed by the Earth Observation Group at the Payne Institute, then calibrated against country-level reporting.

## Assumptions

- **Emission factor: 2.57 kg CO2e per cubic metre** of gas flared. This is what the World Bank's own global figures imply, pairing 148 bcm in 2023 with 381 Mt CO2e. The same factor is asserted in the test suite so it cannot drift.
- **Energy: 10.55 kWh thermal per cubic metre at 38% plant efficiency**, so about 4.0 kWh of electricity per cubic metre.
- **Household use: 250 kWh per year.** This is an assumption, not a measurement. Raising it lowers the homes figure; at 500 kWh per home the 2025 total becomes 53 million homes.
- **A year with no detection means no flaring.** Usually true, but thick cloud or a low plume can hide a flare. A shutdown flag is a prompt to check, not a finding.

## Honest limitations

- **The annual cross-check is not independent.** Both files come from the same World Bank edition, so agreement to within 1e-13 percent confirms that no rows were lost when filtering to Nigeria. It says nothing about whether the underlying satellite estimates are correct.
- **Data is annual.** Both public sources publish one row per site per year. There is no monthly detection detail, so seasonality cannot be modelled.
- **Operator names are context only.** They ship with the World Bank metadata and are never used for matching, grouping or attribution.
- **The model is symmetric.** Isolation Forest flags unusual site-years in both directions, so a large operator flaring slightly below its usual rate can appear alongside a genuine spike. The rule flags are the more precise signal.

## Licence and credit

Code in this repository is yours to use. The underlying data belongs to the World Bank, the Earth Observation Group, geoBoundaries and OpenStreetMap, each under the licence listed above. Credit the Earth Observation Group and the World Bank if you reuse the flare figures.