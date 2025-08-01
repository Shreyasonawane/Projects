# Drone Traffic & Road Monitoring Project 

# Features
- Object detection using YOLOv8
- Vehicle tracking using Deep SORT
- Streamlit dashboard for video playback
- Ready to process VisDrone aerial videos

# Project Structure
- `main.py` – detection + tracking logic
- `dashboard/app.py` – visual dashboard
- `models/yolo` – YOLO weights (`.pt`)
- `datasets/VisDrone` – drone videos
- `tracking/` – Deep SORT
- `utils/` – helper functions

#How to Run
```bash
pip install -r requirements.txt
python main.py
streamlit run dashboard/app.py
