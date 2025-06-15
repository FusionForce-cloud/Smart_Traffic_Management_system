import streamlit as st
import json
import torch
import time
from PIL import Image
import numpy as np
import cv2
import easyocr
import matplotlib.pyplot as plt
import os

# === Streamlit App Title ===
st.title("🚦 Smart Traffic Signal System")

# === Load Weather Data ===
st.header("🌦️ Weather Report")
with open("data/simulated_weather_chandigarh.json", "r") as f:
    weather_data = json.load(f)

condition = weather_data['weather'][0]['description'].lower()
temp = weather_data['main']['temp']

if "rain" in condition or "storm" in condition:
    signal_strategy = "Longer green/red cycles for slower traffic"
    weather_condition = "rain"
elif "fog" in condition or "smoke" in condition:
    signal_strategy = "Activate flashing amber signals"
    weather_condition = "fog"
else:
    signal_strategy = "Normal timing"
    weather_condition = "clear"

st.write(f"Condition: {condition}")
st.write(f"Temperature: {temp}°C")
st.write(f"Signal Strategy: {signal_strategy}")

# === Load YOLOv5 Model ===
st.header("📷 Vehicle Detection")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

image_paths = {
    "Phase 1": "images/phase1.png",
    "Phase 2": "images/phase2.png",
    "Phase 3": "images/phase3.png",
    "Phase 4": "images/phase4.png"
}

vehicle_counts = {}
st.subheader("🚗 Detected Vehicles")
for phase, path in image_paths.items():
    img = Image.open(path)
    results = model(img)
    detected_classes = results.pred[0][:, -1]
    count = sum(1 for c in detected_classes if int(c) in [2, 3, 5, 7])
    vehicle_counts[phase] = count
    st.write(f"{phase}: {count} vehicles")

# === Emergency Detection Functions ===
reader = easyocr.Reader(['en'], gpu=False)

def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (300, 100))
    _, thresh = cv2.threshold(resized, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

def cnn_emergency_classifier(crop_img):
    return False  # Placeholder

def ocr_emergency_classifier(crop_img):
    processed_img = preprocess_for_ocr(crop_img)
    results = reader.readtext(processed_img)
    for _, text, _ in results:
        if 'ambulance' in text.lower() or 'emergency' in text.lower():
            return True
    return False

def emergency_vehicle_classifier(crop_img):
    if cnn_emergency_classifier(crop_img):
        return True
    if ocr_emergency_classifier(crop_img):
        return True
    return False

# === Detect Emergency Vehicles ===
st.subheader("🚨 Emergency Vehicle Detection")
emergency_detected_phases = []
os.makedirs("crops", exist_ok=True)
for phase, path in image_paths.items():
    img = Image.open(path)
    results = model(img)
    image_np = np.array(img)
    for i, det in enumerate(results.pred[0]):
        x1, y1, x2, y2, conf, cls_id = det
        if int(cls_id) in [2, 3, 5, 7]:
            crop = image_np[int(y1):int(y2), int(x1):int(x2)]
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            if emergency_vehicle_classifier(crop_bgr):
                emergency_detected_phases.append(phase)
                st.warning(f"🚨 Emergency vehicle detected in {phase}")
                break

# === Allocate Green Times ===
def allocate_green_times(vehicle_counts, weather_condition="clear", total_cycle_time=80, yellow=4, min_green=8):
    total_yellow = yellow * len(vehicle_counts)
    available_green = total_cycle_time - total_yellow
    total_vehicles = sum(vehicle_counts.values())

    green_times = {}
    for phase, count in vehicle_counts.items():
        green = max(min_green, (count / total_vehicles) * available_green) if total_vehicles else available_green / len(vehicle_counts)
        if weather_condition == "rain": green += 5
        elif weather_condition == "fog": green *= 1.2
        green_times[phase] = min(int(green), 60)
    return green_times

green_times = allocate_green_times(vehicle_counts, weather_condition)

# === Traffic Light Strategy ===
st.subheader("🔁 Signal Timings")
if emergency_detected_phases:
    for phase in emergency_detected_phases:
        st.success(f"✅ GREEN for Emergency Phase: {phase} (15s)")
else:
    phase_pairs = [("Phase 1", "Phase 3"), ("Phase 2", "Phase 4")]
    for pair in phase_pairs:
        p1, p2 = pair
        green_time = max(green_times[p1], green_times[p2])
        yellow_time = 4
        red_time = sum(green_times[p] + yellow_time for p in green_times if p not in pair)

        st.write(f"\n🔁 {p1} & {p2} GREEN")
        st.write(f"🟢 GREEN: {green_time}s")
        st.write(f"🟡 YELLOW: {yellow_time}s")
        st.write(f"🔴 RED: {red_time}s for others")
