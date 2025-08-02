import streamlit as st
import os

# Set the title of the dashboard web app
st.title("🚁 Drone Traffic & Road Monitoring Dashboard")

# Define the path to the output video file
video_file = "../output.avi"

# Check if the video file exists
if os.path.exists(video_file):
    # If it exists, display the video on the Streamlit dashboard
    st.video(video_file)
else:
    # If not, show a warning message to the user
    st.warning("Output video not found. Run `main.py` first.")
