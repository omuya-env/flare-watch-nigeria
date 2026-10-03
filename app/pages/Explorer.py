"""Explorer page: filter flare sites by state, operator, size, age and anomaly type."""

import pandas as pd
import pydeck as pdk
import streamlit as st

from utils import load_anomalies, load_sites

st.set_page_config(page_title="Explorer", page_icon="🔎", layout="wide")

sites = load_sites()
if sites.empty:
    st.error("No processed data found. Run the pipeline first.")
    st.stop()

anomalies = load_anomalies()

st.title("Explorer")
st.caption("Filter Nigerian flare sites by state, operator, size, how long they have been burning, and what the model flagged.")

col_state, col_operator, col_level = st.columns(3)
states = sorted(sites["state"].dropna().unique())
operators = sorted(sites["operator"].dropna().unique())
levels = sorted(sites["flare_level"].dropna().unique())

pick_state = col_state.multiselect("State", states, default=[])
pick_operator = col_operator.multiselect("Operator", operators, default=[])
pick_level = col_level.multiselect("Flare level", levels, default=[])

col_year, col_anomaly = st.columns(2)
min_bcm = col_year.slider("Minimum total volume (bcm)", 0.0, float(sites["total_bcm"].max()), 0.0, step=0.01)
if not anomalies.empty and "rule_tags" in anomalies:
    all_rules = sorted({tag.split(":")[0] for tags in anomalies["rule_tags"].dropna() for tag in [tags]})
    rule_labels = sorted({tag.split(": ")[0] for tags in anomalies["rule_tags"].dropna() for tag in [tags]})
    pick_rule = col_anomaly.multiselect("Anomaly type", rule_labels, default=[])
else:
    pick_rule = []

filtered = sites.copy()
if pick_state:
    filtered = filtered[filtered["state"].isin(pick_state)]
if pick_operator:
    filtered = filtered[filtered["operator"].isin(pick_operator)]
if pick_level:
    filtered = filtered[filtered["flare_level"].isin(pick_level)]
filtered = filtered[filtered["total_bcm"] >= min_bcm]

flagged_ids = set()
if pick_rule and not anomalies.empty:
    mask = anomalies["rule_tags"].fillna("").apply(lambda tags: any(rule in tags for rule in pick_rule))
    flagged_ids = set(anomalies.loc[mask, "id"])
    filtered = filtered[filtered["id"].isin(flagged_ids)]

st.write(f"**{len(filtered):,} sites** match the current filters.")

if not filtered.empty:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Map")
        view = pdk.ViewState(latitude=9.08, longitude=8.68, zoom=5.6, pitch=0)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=filtered,
            get_position=["lon", "lat"],
            get_radius="total_bcm",
            radius_min_pixels=800,
            radius_max_pixels=40000,
            get_fill_color=[59, 130, 246],
            get_line_color="[255, 255, 255, 140]",
            pickable=True,
        )
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip="Flare site"))

    with right:
        st.subheader("Cumulative volume by state")
        by_state = filtered.groupby("state", dropna=False)["total_bcm"].sum().sort_values(ascending=False)
        if not by_state.empty:
            st.bar_chart(by_state.rename("bcm flared"))

    st.subheader("Sites")
    columns = [
        "id", "state", "operator", "field_name", "field_type", "flare_level",
        "first_year", "last_year", "active_years", "total_bcm", "peak_bcm",
        "nearest_settlement", "nearest_settlement_km",
    ]
    display = filtered[[c for c in columns if c in filtered.columns]].copy()
    display["total_bcm"] = display["total_bcm"].round(4)
    st.dataframe(
        display.sort_values("total_bcm", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "total_bcm": st.column_config.NumberColumn("Total bcm", format="%.4f"),
            "nearest_settlement_km": st.column_config.NumberColumn("Nearest town (km)", format="%.1f"),
        },
    )
    st.download_button("Download filtered sites as CSV", display.to_csv(index=False).encode("utf-8"), file_name="flare_sites_filtered.csv", mime="text/csv")
else:
    st.info("No sites match these filters. Try widening them.")