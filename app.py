# app.py - Streamlit dashboard for Steam AXIA
# Usage: streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# -------------------------
# CONFIG: paths to your CSVs
# -------------------------
# TS_CSV = "/kaggle/working/outputs/cleaned_bbc_step3.csv"
# EVENTS_CSV = "/kaggle/working/outputs/blowdown_events_clean.csv"
# If using Colab, replace paths with:
TS_CSV = "data/cleaned_bbc_step3.csv"
EVENTS_CSV = "data/events_table_for_presentation.csv" # Corrected path to load enriched events

# -------------------------
# HELPERS
# -------------------------
@st.cache_data
def load_data(ts_path: str, events_path: str):
    ts = pd.read_csv(ts_path, low_memory=False, parse_dates=["Timestamp"])
    events = pd.read_csv(events_path, low_memory=False, parse_dates=["start","end"])
    # ensure timezone-aware datetimes if strings include offsets
    ts["Timestamp"] = pd.to_datetime(ts["Timestamp"], errors="coerce")
    events["start"] = pd.to_datetime(events["start"], errors="coerce")
    events["end"] = pd.to_datetime(events["end"], errors="coerce")
    return ts, events

def event_centered_df(ts, events, fw_col="FeedWater TDS", drum_col_candidate=None, window_min=120):
    # choose drum column
    drum_col = drum_col_candidate
    if drum_col is None:
        # Explicitly look for 'Drum Water  TDS' first, then fallback to general 'Drum' columns
        if 'Drum Water  TDS' in ts.columns:
            drum_col = 'Drum Water  TDS'
        else:
            drum_col = next((c for c in ts.columns if 'Drum' in c or 'DrumWater' in c or 'Drum Water' in c), None)
    aligned_fw = []
    aligned_drum = []
    valid_events = []
    for _, ev in events.iterrows():
        st = ev["start"]
        if pd.isna(st):
            continue
        wstart = st - pd.Timedelta(minutes=window_min)
        wend = st + pd.Timedelta(minutes=window_min)
        window = ts[(ts["Timestamp"] >= wstart) & (ts["Timestamp"] <= wend)].copy()
        if window.empty:
            continue
        rel_minutes = ((window["Timestamp"] - st) / pd.Timedelta(minutes=1)).round().astype(int).values
        if fw_col in window.columns:
            aligned_fw.append(pd.Series(window[fw_col].values, index=rel_minutes))
        if drum_col and drum_col in window.columns:
            aligned_drum.append(pd.Series(window[drum_col].values, index=rel_minutes))
        valid_events.append(st)
    # convert to DataFrames (rows=relative minute, cols=events)
    def to_df(lst):
        if not lst:
            return pd.DataFrame()
        df_align = pd.DataFrame(lst).T
        full_idx = list(range(-window_min, window_min+1))
        df_align = df_align.reindex(full_idx)
        return df_align
    return to_df(aligned_fw), to_df(aligned_drum), drum_col

# -------------------------
# APP LAYOUT
# -------------------------
st.set_page_config(layout="wide", page_title="Steam AXIA Dashboard")
st.title("Steam AXIA — Blowdown & TDS Dashboard")
st.markdown("Interactive dashboard to monitor feedwater/drum TDS, blowdown events, and key metrics.")

# Load data
with st.spinner("Loading data..."):
    ts, events = load_data(TS_CSV, EVENTS_CSV)

# Basic data checks
col1, col2, col3 = st.columns([1,1,1])
col1.metric("Rows (timeseries)", f"{len(ts):,}")
col2.metric("Events detected", f"{len(events):,}")
ts_min = ts["Timestamp"].min()
ts_max = ts["Timestamp"].max()
col3.metric("Time span", f"{ts_min} → {ts_max}")

# sidebar controls
st.sidebar.header("Controls")
horizon = st.sidebar.slider("Event-centered window (minutes each side)", 30, 360, 120, step=30)
fw_col = st.sidebar.selectbox("FeedWater TDS column", options=[c for c in ts.columns if 'Feed' in c or 'FeedWater' in c], index=0)
# find drum column candidates
drum_options = [c for c in ts.columns if 'Drum' in c or 'DrumWater' in c or 'Drum Water' in c]
# Ensure 'Drum Water  TDS' is prioritized if it exists
if 'Drum Water  TDS' in ts.columns:
    default_drum_col_index = drum_options.index('Drum Water  TDS') if 'Drum Water  TDS' in drum_options else 0
else:
    default_drum_col_index = 0
drum_col = st.sidebar.selectbox("Drum column (for TDS)", options=drum_options, index=default_drum_col_index)
st.sidebar.write("Paths used:")
st.sidebar.code(TS_CSV)
st.sidebar.code(EVENTS_CSV)

# -------------------------
# Time series with event markers
# -------------------------
st.subheader("TDS Timeline with Blowdown Events")
# sample last N hours selector
last_hours = st.selectbox("Show last N hours", [24, 48, 72, 168, int((ts_max - ts_min)/pd.Timedelta(hours=1))], index=2)
end = ts["Timestamp"].max()
start = end - pd.Timedelta(hours=last_hours)
ts_window = ts[(ts["Timestamp"] >= start) & (ts["Timestamp"] <= end)]

fig = go.Figure()
if fw_col in ts_window.columns:
    fig.add_trace(go.Scatter(x=ts_window["Timestamp"], y=ts_window[fw_col], name="FeedWater TDS", mode="lines"))
if drum_col and drum_col in ts_window.columns:
    fig.add_trace(go.Scatter(x=ts_window["Timestamp"], y=ts_window[drum_col], name="Drum TDS", mode="lines"))
# add blowdown event vertical lines
evs = events[(events["start"] >= start) & (events["start"] <= end)]
for st_event in evs["start"].dropna():
    fig.add_vline(x=st_event, line=dict(color="red", width=1), opacity=0.6)
fig.update_layout(height=350, xaxis_title="Time", yaxis_title="TDS (ppm)")
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Event-centered TDS plot
# -------------------------
st.subheader("Event-Centered Mean TDS (± minutes)")
fw_df, drum_df, drum_col_detected = event_centered_df(ts, events, fw_col=fw_col, drum_col_candidate=drum_col, window_min=horizon)
if fw_df.empty and drum_df.empty:
    st.info("No events have enough surrounding data for event-centered plot.")
else:
    traces = []
    if not fw_df.empty:
        mean_fw = fw_df.mean(axis=1)
        sem_fw = fw_df.sem(axis=1).fillna(0)
        x = mean_fw.index
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=mean_fw.values, name="FeedWater TDS (mean)", mode="lines"))
        fig2.add_trace(go.Scatter(x=x, y=(mean_fw+1.96*sem_fw).values, name="FW TDS upper CI", mode="lines", line=dict(dash="dash"), opacity=0.3))
        fig2.add_trace(go.Scatter(x=x, y=(mean_fw-1.96*sem_fw).values, name="FW TDS lower CI", mode="lines", line=dict(dash="dash"), opacity=0.3))
    else:
        fig2 = go.Figure()
    if not drum_df.empty:
        mean_dr = drum_df.mean(axis=1)
        sem_dr = drum_df.sem(axis=1).fillna(0)
        fig2.add_trace(go.Scatter(x=mean_dr.index, y=mean_dr.values, name=f"{drum_col_detected} (mean)", mode="lines"))
        fig2.add_trace(go.Scatter(x=mean_dr.index, y=(mean_dr+1.96*sem_dr).values, name="Drum upper CI", mode="lines", line=dict(dash="dash"), opacity=0.3))
        fig2.add_trace(go.Scatter(x=mean_dr.index, y=(mean_dr-1.96*sem_dr).values, name="Drum lower CI", mode="lines", line=dict(dash="dash"), opacity=0.3))
    fig2.add_vline(x=0, line=dict(color="red", dash="dash"), annotation_text="Blowdown start", annotation_position="top left")
    fig2.update_layout(height=380, xaxis_title="Minutes relative to blowdown start", yaxis_title="TDS (ppm)")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Inter-blowdown time by FW TDS groups
# -------------------------
st.subheader("Inter-blowdown interval vs FeedWater TDS")
# show grouping options
grouping = st.selectbox("Group by", ["Median split", "Domain threshold (choose below)"])
threshold = None
if grouping.startswith("Domain"):
    threshold = st.number_input("High TDS threshold (ppm)", value=100)
# prepare grouping
events_en = events.copy()
if grouping == "Median split":
    med = events_en['pre60_mean_tds'].median()
    events_en['fw_group'] = np.where(events_en['pre60_mean_tds'] > med, "High", "Low")
else:
    events_en['fw_group'] = np.where(events_en['pre60_mean_tds'] >= threshold, "High", "Low")
# remove NaNs
plot_events = events_en.dropna(subset=['time_to_next_min','pre60_mean_tds'])
if plot_events.empty:
    st.info("Not enough events with pre60 TDS and time_to_next for this analysis.")
else:
    fig3 = px.box(plot_events, x='fw_group', y='time_to_next_min', points="all", labels={'time_to_next_min': 'Time to next blowdown (min)', 'fw_group':'FW TDS group'})
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)
    # show medians table
    med_tab = plot_events.groupby('fw_group')['time_to_next_min'].median().reset_index().rename(columns={'time_to_next_min':'median_min'})
    st.table(med_tab)

# -------------------------
# Blowdown effectiveness table & bar chart
# -------------------------
st.subheader("Blowdown effectiveness")
cols_to_show = ['start','end','duration_min','volume_blw','pre60_mean_tds','pre60_mean_drum','post60_mean_drum','delta_removed_ppm']
present = [c for c in cols_to_show if c in events.columns]
st.dataframe(events[present].sort_values('start', ascending=False).reset_index(drop=True), height=300)
# bar chart
if 'delta_removed_ppm' in events.columns:
    fig4 = px.bar(events.reset_index().rename(columns={'index':'event_index'}), x='event_index', y='delta_removed_ppm', labels={'delta_removed_ppm':'TDS removed (ppm)', 'event_index':'Event index'})
    fig4.update_layout(height=300)
    st.plotly_chart(fig4, use_container_width=True)

# -------------------------
# Export / download
# -------------------------
st.markdown("---")
st.subheader("Export")
st.write("Download cleaned CSV and events CSV for external use:")
st.markdown(f"- Cleaned time-series: `{TS_CSV}`")
st.markdown(f"- Events table: `{EVENTS_CSV}`")
if st.button("Download events CSV"):
    st.download_button("Download events CSV", data=events.to_csv(index=False).encode('utf-8'), file_name="events_table_for_presentation.csv", mime="text/csv")

st.info("Tip: run `streamlit run app.py` in your terminal. In Colab you can use ngrok or run locally and forward a port.")
