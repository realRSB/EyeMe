import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from glob import glob
import os
from datetime import datetime
import time
import shutil
import cv2
import mediapipe as mp

from modules.train_face_recognizer import train_eye_recognizer

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
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
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
file = "logs/eye_health_log.csv"
try:
    df = pd.read_csv(file)
except FileNotFoundError:
    st.error("No data found in logs/. Run the main app to log some eye health data.")
    st.stop()

# ------------------- RETRAIN MODEL --------------------
st.subheader("🔁 Face Recognizer Control")
if st.button("Retrain Face Recognizer"):
    with st.spinner("Retraining model on eye snapshots..."):
        success = train_eye_recognizer()
        if success:
            st.success("Model retrained successfully!")
        else:
            st.error("Training failed. No data found in `face_training/`.")

st.subheader("➕ Add New User")

new_user = st.text_input("Enter new user name")

if new_user:
    max_photos = 20
    num_photos = st.slider("How many snapshots to capture?", min_value=1, max_value=max_photos, value=3)

    if st.button("Capture Snapshots for New User"):
        os.makedirs(f"face_training/{new_user}", exist_ok=True)

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

        cap = cv2.VideoCapture(0)
        st.info(f"📸 Preparing to capture {num_photos} snapshots...")

        with st.spinner("⏳ Starting in 3 seconds..."):
            time.sleep(3)

        st.info(f"🔄 Capturing now... {num_photos} total")
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

                img_h, img_w = frame.shape[:2]
                eye_landmarks = LEFT_EYE + RIGHT_EYE
                xs = [int(landmarks[i].x * img_w) for i in eye_landmarks]
                ys = [int(landmarks[i].y * img_h) for i in eye_landmarks]
                x_min, x_max = max(min(xs) - 10, 0), min(max(xs) + 10, img_w)
                y_min, y_max = max(min(ys) - 10, 0), min(max(ys) + 10, img_h)

                eye_crop = frame[y_min:y_max, x_min:x_max]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"face_training/{new_user}/{timestamp}_{i}.jpg"
                cv2.imwrite(filepath, eye_crop)
                success_count += 1

            st.info(f"✅ Captured image {i+1} of {num_photos}")
            time.sleep(1)  # Wait 1 second between captures

        cap.release()
        st.success(f"🎉 Saved {success_count} snapshots for '{new_user}'")

# ------------------- SNAPSHOT GALLERY --------------------
st.subheader("📸 Eye Snapshot Gallery (by Person)")
snapshot_files = sorted(glob("snapshots/*.jpg"))

gallery = {}
for file in snapshot_files:
    filename = os.path.basename(file)
    if "_" in filename:
        name = filename.split("_")[0]
        gallery.setdefault(name, []).append(file)

for name, files in gallery.items():
    st.markdown(f"### 👤 {name.capitalize()}")
    cols = st.columns(3)
    for i, file in enumerate(files):
        with cols[i % 3]:
            st.image(file, caption=os.path.basename(file), width=200)

st.subheader("🧑‍🤝‍🧑 Enrolled Users")

users = sorted([d for d in os.listdir("face_training") if os.path.isdir(os.path.join("face_training", d))])
if users:
    for user in users:
        st.markdown(f"- 👁️ **{user}** ({len(os.listdir(os.path.join('face_training', user)))} images)")
else:
    st.info("No users enrolled yet.")

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

# ------------------- DELETE / RENAME USERS --------------------
st.subheader("🗑 Delete a User")
user_to_delete = st.selectbox("Select a user to delete", users, key="delete_user")
if st.button("Delete User"):
    try:
        shutil.rmtree(f"face_training/{user_to_delete}")
        st.success(f"Deleted user '{user_to_delete}'")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to delete: {e}")

st.subheader("✏️ Rename a User")
user_to_rename = st.selectbox("Select a user to rename", users, key="rename_user")
new_name = st.text_input("Enter new name", key="rename_input")
if st.button("Rename User"):
    try:
        os.rename(f"face_training/{user_to_rename}", f"face_training/{new_name}")
        st.success(f"Renamed '{user_to_rename}' to '{new_name}'")
        st.rerun()
    except Exception as e:
        st.error(f"Rename failed: {e}")

# ------------------- DOWNLOAD --------------------
st.subheader("📂 Export")
try:
    with open(file, "rb") as f:
        st.download_button("Download Full Eye Health CSV Log", data=f, file_name="eye_health_log.csv")
except FileNotFoundError:
    st.warning("CSV file not found in logs/.")
