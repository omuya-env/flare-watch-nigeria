"""Site detail page: one flare site's history, flags with explanations, and carbon contribution."""

import ast

import pandas as pd
import pydeck as pdk
import streamlit as st

from utils import load_anomalies, load_co2e, load_sites

st.set_page_config(page_title="Site detail", page_icon="📍", layout="wide")

sites = load_sites()
if sites.empty:
    st.error("No processed data found. Run the pipeline first.")
    st.stop()

anomalies = load_anomalies()
co2e = load_co2e()

query_id = st.query_params.get("site")
if isinstance(query_id, list):
    query_id = query_id[0]

def split_tags(value) -> list[str]:
    """Read the stored rule tag list back out of the CSV cell."""
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return [str(value)]
    return parsed if isinstance(parsed, list) else [str(value)]


ordered = sites.sort_values("total_bcm", ascending=False)
id_to_label = {
    row.id: f"{row.id} | {row.state or 'unknown state'} | {row.operator or 'unknown operator'} | {row.total_bcm:.3f} bcm"
    for row in ordered.itertuples()
}

default_index = list(id_to_label).index(query_id) if query_id in id_to_label else 0
choice = st.selectbox("Flare site", list(id_to_label), index=default_index, format_func=lambda value: id_to_label[value])
st.query_params["site"] = choice

site = ordered[ordered["id"] == choice].iloc[0]
site_co2e = co2e[co2e["id"] == choice] if not co2e.empty else pd.DataFrame()
site_anomalies = anomalies[anomalies["id"] == choice] if not anomalies.empty else pd.DataFrame()

st.title(f"Flare site {site['id']}")
st.caption(f"{site['state'] or 'State unknown'} | nearest community {site['nearest_settlement'] or 'unknown'} at {site['nearest_settlement_km']:.1f} km" if pd.notna(site.get("nearest_settlement_km")) else f"{site['state'] or 'State unknown'}")

info_left, info_mid, info_right = st.columns(3)
info_left.markdown(
    f"**Operator (display only)**\n\n{site['operator'] if pd.notna(site['operator']) else 'not published'}\n\n"
    f"Field: {site['field_name'] if pd.notna(site['field_name']) else 'not published'}"
)
info_mid.markdown(
    f"**Site profile**\n\n"
    f"First seen: {int(site['first_year'])}\n\n"
    f"Last seen: {int(site['last_year'])}\n\n"
    f"Active years: {int(site['active_years'])} of {int(site['active_span_years'])}"
)
info_right.markdown(
    f"**Volume**\n\n"
    f"Cumulative: {site['total_bcm']:.4f} bcm\n\n"
    f"Peak: {site['peak_bcm']:.4f} bcm in {int(site['peak_year'])}\n\n"
    f"Flare level: {site['flare_level'] if pd.notna(site['flare_level']) else 'unknown'}"
)

trend_left, trend_right = st.columns([2, 1])

with trend_left:
    st.subheader("Annual flare volume")
    if not site_co2e.empty:
        st.line_chart(site_co2e.set_index("year")["BCM"].astype(float).rename("bcm flared"))
    st.caption("Billion cubic metres per year. A year with no bar or point means the satellite saw no flare at this site.")

with trend_right:
    st.subheader("Satellite detection frequency")
    if not site_co2e.empty and site_co2e["detection_freq"].notna().any():
        st.line_chart(site_co2e.dropna(subset=["detection_freq"]).set_index("year")["detection_freq"].astype(float).rename("detection frequency"))
        st.caption("Share of clear-sky passes in which VIIRS saw the flare. This is the persistence signal.")
    else:
        st.write("No EOG detection frequency for this site.")

st.subheader("Flagged years")
if site_anomalies.empty:
    st.success("No anomalies flagged for this site.")
else:
    for row in site_anomalies.sort_values("year").itertuples():
        st.markdown(f"**{int(row.year)}** &nbsp;|&nbsp; anomaly score {row.anomaly_score:.2f} &nbsp;|&nbsp; flagged by {row.flag_source}")
        if pd.notna(row.rule_tags) and row.rule_tags:
            for tag in split_tags(row.rule_tags):
                st.markdown(f"- {tag}")
        if "model" in str(row.flag_source):
            st.caption(f"Why the model flagged it: {row.shap_explanation}")

st.subheader("Carbon contribution")
if not site_co2e.empty:
    total_tonnes = float(site_co2e["co2e_tonnes"].sum())
    st.metric("Cumulative CO2e", f"{total_tonnes / 1e6:.2f} Mt")
    st.line_chart(site_co2e.set_index("year")["co2e_tonnes"].astype(float).rename("tonnes CO2e per year"))
    if not co2e.empty:
        share = total_tonnes / float(co2e["co2e_tonnes"].sum()) * 100
        st.caption(f"That is {share:.3f}% of all CO2e attributed to Nigerian flare sites in this dataset.")

st.subheader("Location")
if pd.notna(site["lat"]) and pd.notna(site["lon"]):
    view = pdk.ViewState(latitude=float(site["lat"]), longitude=float(site["lon"]), zoom=10, pitch=0)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame([{"lat": site["lat"], "lon": site["lon"], "total_bcm": site["total_bcm"]}]),
        get_position=["lon", "lat"],
        get_radius="total_bcm",
        radius_min_pixels=1500,
        radius_max_pixels=25000,
        get_fill_color=[239, 68, 68],
    )
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip="Flare site"))