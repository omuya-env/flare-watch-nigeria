"""National view: headline figures, an anomaly map sized by flare volume, and the worst flagged sites."""

import pandas as pd
import pydeck as pdk
import streamlit as st

from utils import load_anomalies, load_national, load_sites, load_yearly

st.set_page_config(page_title="Flare Watch Nigeria", page_icon="🔥", layout="wide")

sites = load_sites()
anomalies = load_anomalies()
national = load_national()

if sites.empty:
    st.error("No processed data found. Run the pipeline first: scripts/download_data.py, build_sites.py, build_timeseries.py, estimate_co2e.py, then models/train_anomaly.py")
    st.stop()

summary = national.get("latest_year", {})
cumulative = national.get("cumulative", {})
peak = national.get("peak_year", {})

st.title("Flare Watch Nigeria")
st.caption(f"Satellite-based gas flaring tracker for Nigeria, {national.get('coverage', {}).get('first_year', '')} to {national.get('coverage', {}).get('last_year', '')}")

st.markdown(
    f"Nigeria flared **{summary.get('bcm', 0):.2f} billion cubic metres of gas in {summary.get('year', '')}**, "
    f"about **{summary.get('co2e_mt', 0):.1f} million tonnes of CO2 equivalent**. "
    f"At {national.get('plant_efficiency', 0):.0%} plant efficiency that is roughly **{summary.get('homes_equivalent', 0) / 1e6:.0f} million homes** worth of electricity for a year."
)

latest_flagged = anomalies[anomalies["year"] == anomalies["year"].max()] if not anomalies.empty else pd.DataFrame()
ever_flagged = anomalies["id"].nunique() if not anomalies.empty else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Flared in latest year", f"{summary.get('bcm', 0):.2f} bcm")
col2.metric("CO2e in latest year", f"{summary.get('co2e_mt', 0):.1f} Mt")
col3.metric("Active flare sites", f"{summary.get('active_sites', 0):,}")
col4.metric(f"Sites flagged in {int(anomalies['year'].max()) if not anomalies.empty else ''}", f"{latest_flagged['id'].nunique():,}" if not latest_flagged.empty else "0")

if peak:
    st.info(f"Peak recorded year: {peak.get('year')} at {peak.get('bcm', 0):.2f} bcm, {peak.get('co2e_mt', 0):.1f} Mt CO2e.")

st.subheader("Flare sites and anomaly flags")
st.caption("Circle size is the flared volume in that year. Red means the year was flagged by the model or by a rule.")

if anomalies.empty:
    map_frame = sites.assign(volume_bcm=sites["peak_bcm"], status="No flag")
else:
    recent = anomalies[anomalies["year"] >= anomalies["year"].max() - 2]
    map_frame = sites.merge(
        recent.groupby("id").agg(anomaly_score=("anomaly_score", "max"), flag_source=("flag_source", "last"), flag_year=("year", "max")),
        on="id",
        how="left",
    )
    map_frame["status"] = map_frame["flag_source"].fillna("No flag")
    map_frame["volume_bcm"] = map_frame["peak_bcm"]
    map_frame["anomaly_score"] = map_frame["anomaly_score"].fillna(0.0)

st.caption(f"{ever_flagged:,} of {len(sites):,} sites have been flagged at least once since {national.get('coverage', {}).get('first_year', '')}.")

view = pdk.ViewState(latitude=9.08, longitude=8.68, zoom=5.6, pitch=0)
flagged_map = map_frame[map_frame["status"] != "No flag"]
normal_map = map_frame[map_frame["status"] == "No flag"]
def make_layer(frame, colour: list[int], layer_id: str):
    """One deck.gl circle layer; separate layers keep the colours as literals."""
    return pdk.Layer(
        "ScatterplotLayer",
        data=frame,
        id=layer_id,
        get_position=["lon", "lat"],
        get_radius="volume_bcm",
        radius_min_pixels=800,
        radius_max_pixels=40000,
        get_fill_color=colour,
        get_line_color=[255, 255, 255, 140],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
    )


layers = [
    make_layer(normal_map, [59, 130, 246], "normal"),
    make_layer(flagged_map, [239, 68, 68], "flagged"),
]
st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip="Flare site"))
st.caption(f"{len(normal_map):,} sites with no flag, {len(flagged_map):,} with at least one flagged year.")

left, right = st.columns([1, 1])

with left:
    st.subheader("Sites with the most flags")
    if not anomalies.empty:
        counts = anomalies.groupby("id").size().sort_values(ascending=False).head(15).rename("flags").reset_index()
        counts = counts.merge(sites[["id", "state", "operator", "field_name"]], on="id", how="left")
        st.dataframe(counts, use_container_width=True, hide_index=True)
    else:
        st.write("No flags recorded.")

with right:
    st.subheader("National volume by year")
    yearly = load_yearly()
    if yearly.empty:
        st.write("No yearly totals found.")
    else:
        st.line_chart(yearly[["year", "bcm"]], x="year", y="bcm")
        st.caption("Billion cubic metres of gas flared per year, all Nigerian sites.")

st.subheader("About this data")
st.markdown(
    "Flare sites and annual volumes come from the World Bank Global Gas Flaring Tracker dataset, which is built from "
    "NOAA VIIRS satellite observations processed by the Earth Observation Group at the Payne Institute. Detection "
    "frequency and flame temperature come from the EOG annual flaring workbooks. Every figure here is a satellite "
    "estimate, not a self-reported meter reading."
)
st.caption("Assumptions, formulas and the full method are documented in EXPLANATION.md.")