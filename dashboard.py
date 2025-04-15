import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from glob import glob
import os
from datetime import datetime

# ------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Eyeris Dashboard", layout="centered", page_icon="😎")

# ------------------- STYLING --------------------
custom_css = """
<style>
body, .stApp {
    background-color: black;
    color: white;
}

h1, h2, h3, h4, h5, h6, .stMarkdown, .stDataFrame, .stMetric, .stNumberInput, .stTextInput {
    color: white !important;
}

/* Hide main menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Center the title */
.stApp > header, .block-container {
    text-align: center !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ------------------- TITLE --------------------
st.title("EYERIS: Eye Health Dashboard")

# ------------------- DARK MODE TOGGLE --------------------
dark_mode = st.toggle("🌚 Enable Dark Mode", value=True)

if not dark_mode:
    st.markdown("""
    <style>
    body, .stApp {
        background-color: white;
        color: black;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stDataFrame, .stMetric, .stNumberInput, .stTextInput {
        color: black !important;
    }
    .stMetric label, .stMetric span {
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------- LOAD DATA --------------------
file = "eye_health_log.csv"
try:
    df = pd.read_csv(file)
except FileNotFoundError:
    st.error("No data found. Run the main app to log some eye health data.")
    st.stop()

# ------------------- EYE SNAPSHOT GALLERY --------------------
st.subheader("📷 Daily Eye Snapshot Gallery")
snapshot_files = sorted(glob("snapshots/eye_*.jpg"))[::-1]

if st.button("🔍 Show Last 7 Days Only"):
    today = datetime.now()
    snapshot_files = [f for f in snapshot_files if (today - datetime.strptime(os.path.basename(f).split("_")[1].split(".")[0], "%Y-%m-%d_%H-%M-%S")).days <= 7]

if st.button("🔍 Reset Eye Snapshot Gallery"):
    for f in glob("snapshots/eye_*.jpg"):
        os.remove(f)
    st.success("Snapshot gallery cleared. Refresh the app to see changes.")
    st.stop()

if snapshot_files:
    cols = st.columns(3)
    for idx, file in enumerate(snapshot_files):
        date_str = os.path.basename(file).split("_")[1].split(".")[0].replace("_", " ")
        with cols[idx % 3]:
            st.image(file, caption=f"{date_str}", width=200, use_container_width=False)
else:
    st.info("No snapshots found yet. Run the main app to capture your daily eye image.")

# ------------------- METRICS --------------------
st.subheader("📊 Overview")
st.metric("Total Runtime (min)", f"{df['time_min'].iloc[-1]:.1f}")
st.metric("Avg Blink Rate", f"{df['blink_rate'].mean():.1f}")
st.metric("Avg Redness", f"{df['redness'].mean():.3f}")
st.metric("Avg Pupil Diameter", f"{df['pupil_diameter'].mean():.2f}")
st.metric("Most Frequent Strain Level", df['strain_level'].mode()[0])

# ------------------- SPEEDOMETER GAUGE --------------------
latest = df.iloc[-1]
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=latest['health_score'],
    title={'text': "Eye Health Score"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "white"},
        'bgcolor': "black" if dark_mode else "white",
        'borderwidth': 2,
        'bordercolor': "white" if dark_mode else "black",
        'steps': [
            {'range': [0, 50], 'color': "#8B0000"},
            {'range': [50, 75], 'color': "#FFD700"},
            {'range': [75, 100], 'color': "#00FF7F"},
        ]
    }
))
st.plotly_chart(fig, use_container_width=True)

# ------------------- TREND CHARTS --------------------
st.subheader("🔢 Trends")
st.line_chart(df.set_index('time_min')[['blink_rate']], height=250, use_container_width=True)
st.caption("Blink Rate Over Time")
st.line_chart(df.set_index('time_min')[['redness']], height=250, use_container_width=True)
st.caption("Redness Over Time")
st.line_chart(df.set_index('time_min')[['pupil_diameter']], height=250, use_container_width=True)
st.caption("Pupil Diameter Over Time")
st.line_chart(df.set_index('time_min')[['health_score']], height=250, use_container_width=True)
st.caption("Health Score Over Time")

# ------------------- DOWNLOAD LOG --------------------
st.subheader("📂 Export")
try:
    with open(file, "rb") as f:
        st.download_button("Download Full Eye Health CSV Log", data=f, file_name="eye_health_log.csv")
except FileNotFoundError:
    st.warning("CSV file not found.")