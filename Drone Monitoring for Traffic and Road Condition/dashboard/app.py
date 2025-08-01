import streamlit as st
import os

st.title("🚁 Drone Traffic & Road Monitoring Dashboard")

video_file = "../output.avi"
if os.path.exists(video_file):
    st.video(video_file)
else:
    st.warning("Output video not found. Run `main.py` first.")
