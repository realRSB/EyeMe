import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import cv2
import shutil
import time
import mediapipe as mp
from pathlib import Path

from recognizer.train_face_recognizer import train_eye_recognizer
from utils.snapshot_helpers import crop_and_save_eye_snapshot

# ------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="EYEME Dashboard", layout="centered", page_icon="😎")

# ------------------- DARK MODE --------------------
st.markdown("""
<style>
body, .stApp {
    background-color: black;
    color: white;
}
h1, h2, h3, h4, h5, h6, .stMarkdown, .stDataFrame, .stMetric, .stNumberInput, .stTextInput {
    color: white !important;
}
#MainMenu, footer {visibility: hidden;}
.stApp > header, .block-container { text-align: center !important; }
</style>
""", unsafe_allow_html=True)

st.title("EYEME: Eye Health Dashboard")

# ------------------- PATH SETUP --------------------
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "data" / "logs"
SNAPSHOT_DIR = BASE_DIR / "data" / "snapshots"
TRAINING_DIR = BASE_DIR / "data" / "face_training"

# ------------------- LOAD LOG FILE --------------------
log_files = sorted(LOG_DIR.glob("eye_health_log_*.csv"))
if log_files:
    selected_log = st.selectbox("📁 Select Session Log", [str(f) for f in log_files], index=len(log_files)-1)
    df = pd.read_csv(selected_log)
    st.success(f"Loaded: {Path(selected_log).name}")
else:
    st.error("No logs found.")
    st.stop()

# ------------------- RETRAIN MODEL --------------------
st.subheader("🔁 Face Recognizer Control")
if st.button("Retrain Face Recognizer"):
    with st.spinner("Retraining model on eye snapshots..."):
        success = train_eye_recognizer()
        if success:
            st.success("Model retrained successfully!")
        else:
            st.error("Training failed. No data found.")

# ------------------- ADD NEW USER --------------------
st.subheader("➕ Add New User")
new_user = st.text_input("Enter new user name")
max_photos = 20
num_photos = st.slider("Snapshots to capture", 1, max_photos, 3)

if new_user and st.button("Capture Snapshots for New User"):
    user_dir = TRAINING_DIR / new_user
    user_dir.mkdir(parents=True, exist_ok=True)
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    cap = cv2.VideoCapture(0)

    st.info("⏳ Starting in 3 seconds...")
    time.sleep(3)
    st.info(f"Capturing {num_photos} snapshots...")

    success_count = 0
    for i in range(num_photos):
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            LEFT_EYE = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]
            eye_landmarks = LEFT_EYE + RIGHT_EYE

            saved_path = crop_and_save_eye_snapshot(
                frame, landmarks, eye_landmarks, str(user_dir), f"{new_user}_"
            )
            success_count += 1
            st.info(f"Captured {i+1}/{num_photos}")
            time.sleep(1)

    cap.release()
    st.success(f"✅ Saved {success_count} snapshots for '{new_user}'")

# ------------------- SNAPSHOT GALLERY --------------------
st.subheader("📸 Eye Snapshot Gallery (by Person)")
gallery = {}
for file in sorted(SNAPSHOT_DIR.glob("*.jpg")):
    name = file.name.split("_")[0]
    gallery.setdefault(name, []).append(str(file))

for name, files in gallery.items():
    st.markdown(f"### 👤 {name}")
    cols = st.columns(3)
    for i, file in enumerate(files):
        with cols[i % 3]:
            st.image(file, caption=Path(file).name, use_container_width=True)

# ------------------- ENROLLED USERS --------------------
st.subheader("🧑‍🤝‍🧑 Enrolled Users")
users = sorted([d.name for d in TRAINING_DIR.iterdir() if d.is_dir()])
for user in users:
    count = len(list((TRAINING_DIR / user).glob("*.jpg")))
    st.markdown(f"- 👁️ **{user}** ({count} images)")

# ------------------- DELETE / RENAME --------------------
st.subheader("🗑 Delete a User")
to_delete = st.selectbox("Select user", users, key="delete_user")
if st.button("Delete User"):
    shutil.rmtree(TRAINING_DIR / to_delete)
    st.success(f"Deleted '{to_delete}'")
    st.rerun()

st.subheader("✏️ Rename a User")
to_rename = st.selectbox("Select user", users, key="rename_user")
new_name = st.text_input("New name", key="rename_input")
if st.button("Rename User"):
    os.rename(TRAINING_DIR / to_rename, TRAINING_DIR / new_name)
    st.success(f"Renamed '{to_rename}' to '{new_name}'")
    st.rerun()

# ------------------- METRICS --------------------
st.subheader("📊 Overview")
st.metric("Total Runtime (min)", f"{df['time_min'].iloc[-1]:.1f}")
st.metric("Avg Blink Rate", f"{df['blink_rate'].mean():.1f}")
st.metric("Avg Redness", f"{df['redness'].mean():.3f}")
st.metric("Avg Pupil Diameter", f"{df['pupil_diameter'].mean():.2f}")
st.metric("Most Frequent Strain Level", df['strain_level'].mode()[0])

# ------------------- SPEEDOMETER --------------------
st.subheader("💡 Eye Health Score")
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=df['health_score'].iloc[-1],
    title={'text': "Eye Health Score"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "white"},
        'bgcolor': "black",
        'borderwidth': 2,
        'bordercolor': "white",
        'steps': [
            {'range': [0, 50], 'color': "#8B0000"},
            {'range': [50, 75], 'color': "#FFD700"},
            {'range': [75, 100], 'color': "#00FF7F"},
        ]
    }
))
st.plotly_chart(fig, use_container_width=True)

# ------------------- TRENDS --------------------
st.subheader("📈 Trends")
st.line_chart(df.set_index("time_min")[['blink_rate']])
st.caption("Blink Rate Over Time")
st.line_chart(df.set_index("time_min")[['redness']])
st.caption("Redness Over Time")
st.line_chart(df.set_index("time_min")[['pupil_diameter']])
st.caption("Pupil Diameter Over Time")
st.line_chart(df.set_index("time_min")[['health_score']])
st.caption("Health Score Over Time")

# ------------------- EXPORT --------------------
st.subheader("📂 Export CSV")
with open(selected_log, "rb") as f:
    st.download_button("Download CSV", f, file_name=Path(selected_log).name)
