import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="WSN Monitor", layout="wide")
st.title("WSN Monitor: Faults, Energy and Network Health")

df = pd.read_csv("dashboard_v3.csv", parse_dates=["timestamp"])
df["delivery_2h"] = df.groupby("node_id").delivered.transform(lambda s: s.rolling(120, min_periods=1).mean())

tab1, tab2, tab3 = st.tabs(["Fault detection", "Battery and network", "Latency"])

with tab1:
    node = st.selectbox("Node", sorted(df.node_id.unique()), key="n1")
    sensor = st.selectbox("Sensor", ["temperature", "humidity", "light"])
    system = st.radio("Detector", ["combined", "classical"], horizontal=True,
                        format_func=lambda x: "Classical + Isolation Forest" if x == "combined"
                        else "Classical (ranges + watchdog)")
    d = df[(df.node_id == node) & (df.split == "test")]
    c1, c2, c3 = st.columns(3)
    c1.metric("True faults", int(d.is_fault.sum()))
    c2.metric("Flagged", int(d[system].sum()))
    c3.metric("Caught", int(((d[system] == 1) & (d.is_fault == 1)).sum()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.timestamp, y=d[sensor], mode="lines", name=sensor, line=dict(width=1)))
    fl = d[d[system] == 1]
    fig.add_trace(go.Scatter(x=fl.timestamp, y=fl[sensor], mode="markers", name="flagged",
                                marker=dict(color="red", size=5)))
    ms = d[(d.is_fault == 1) & (d[system] == 0)]
    fig.add_trace(go.Scatter(x=ms.timestamp, y=ms[sensor], mode="markers", name="missed fault",
                                marker=dict(color="orange", size=5, symbol="x")))
    rng = {"temperature": [10, 45], "humidity": [30, 90], "light": [0, 450]}
    fig.update_layout(height=420, yaxis=dict(range=rng[sensor]), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Test split only. Y-axis clipped to the normal range; dead-node gaps appear as breaks in the line.")

with tab2:
    fig = go.Figure()
    for n, g in df.groupby("node_id"):
        fig.add_trace(go.Scatter(x=g.timestamp, y=g.battery_pct, mode="lines", name=f"Node {n}"))
    fig.update_layout(height=320, yaxis_title="Battery %", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    fig = go.Figure()
    for n, g in df.groupby("node_id"):
        fig.add_trace(go.Scatter(x=g.timestamp, y=g.delivery_2h, mode="lines", name=f"Node {n}"))
    fig.update_layout(height=320, yaxis_title="Delivery ratio (2h)", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Nodes 2 and 3 are relays and die first; nodes 4 and 5 lose their path when their relay dies.")

with tab3:
    lat = df.dropna(subset=["latency_ms"])
    fig = go.Figure()
    for h, g in lat.groupby("hops"):
        fig.add_trace(go.Histogram(x=g.latency_ms, name=f"{h}-hop", opacity=0.6, nbinsx=60))
    fig.update_layout(barmode="overlay", height=380, xaxis_title="Latency (ms)",
                        margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(lat.groupby("hops").latency_ms.describe().round(1), width="stretch")