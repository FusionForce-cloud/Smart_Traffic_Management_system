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
from yolov5 import utils  # Assuming yolov5 is properly installed

# Set up Streamlit page
st.set_page_config(page_title="Smart Traffic Signal System", layout="wide")
st.title("🚦 Smart Traffic Signal System with Weather & Emergency Detection")

# Initialize session state
if 'emergency_detected' not in st.session_state:
    st.session_state.emergency_detected = False
if 'emergency_phases' not in st.session_state:
    st.session_state.emergency_phases = []

# Sidebar for controls
with st.sidebar:
    st.header("Configuration")
    total_cycle_time = st.slider("Total Cycle Time (seconds)", 60, 120, 80)
    min_green_time = st.slider("Minimum Green Time (seconds)", 5, 20, 8)
    yellow_time = st.slider("Yellow Time (seconds)", 2, 6, 4)
    weather_condition = st.selectbox(
        "Weather Condition",
        ("Clear", "Rain/Storm", "Fog/Smoke"),
        index=0
    )
    
    # File uploaders
    st.header("Upload Images")
    phase1_img = st.file_uploader("Phase 1 Image", type=["png", "jpg", "jpeg"])
    phase2_img = st.file_uploader("Phase 2 Image", type=["png", "jpg", "jpeg"])
    phase3_img = st.file_uploader("Phase 3 Image", type=["png", "jpg", "jpeg"])
    phase4_img = st.file_uploader("Phase 4 Image", type=["png", "jpg", "jpeg"])
    
    weather_json = st.file_uploader("Weather Data (JSON)", type=["json"])

# Initialize models
@st.cache_resource
def load_models():
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])
    
    return model, reader

model, reader = load_models()

# Image processing functions
def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (300, 100))
    _, thresh = cv2.threshold(resized, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

# Placeholder for CNN logic
def cnn_emergency_classifier(crop_img):
    # TODO: Replace this with actual model logic
    return False

def ocr_emergency_classifier(crop_img, reader):
    processed_img = preprocess_for_ocr(crop_img)
    results = reader.readtext(processed_img)
    for _, text, _ in results:
        if 'ambulance' in text.lower() or 'emergency' in text.lower():
            return True
    return False

def emergency_vehicle_classifier(crop_img, reader):
    if cnn_emergency_classifier(crop_img):
        return True
    if ocr_emergency_classifier(crop_img, reader):
        return True
    return False

# Main processing function
def process_images():
    image_paths = {
        "Phase 1": phase1_img,
        "Phase 2": phase2_img,
        "Phase 3": phase3_img,
        "Phase 4": phase4_img
    }
    
    vehicle_counts = {}
    emergency_phases = []
    
    # Create columns for image display
    cols = st.columns(4)
    
    for i, (phase, img_file) in enumerate(image_paths.items()):
        if img_file is not None:
            img = Image.open(img_file)
            
            # Display image
            with cols[i]:
                st.image(img, caption=phase, use_column_width=True)
            
            # Process with YOLO
            results = model(img)
            detected_classes = results.pred[0][:, -1]
            vehicle_counts[phase] = sum(1 for c in detected_classes if int(c) in [2, 3, 5, 7])
            
            # Check for emergency vehicles
            image_np = np.array(img)
            for j, det in enumerate(results.pred[0]):
                x1, y1, x2, y2, conf, cls_id = det
                if int(cls_id) in [2, 3, 5, 7]:
                    crop = image_np[int(y1):int(y2), int(x1):int(x2)]
                    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    if emergency_vehicle_classifier(crop_bgr, reader):
                        emergency_phases.append(phase)
                        break
            
            st.write(f"{phase}: {vehicle_counts[phase]} vehicles")
        else:
            vehicle_counts[phase] = 0
    
    return vehicle_counts, emergency_phases

# Green time allocation function
def allocate_green_times(vehicle_counts, weather_condition, total_cycle_time, yellow_time, min_green_time):
    total_yellow = yellow_time * len(vehicle_counts)
    available_green = total_cycle_time - total_yellow
    total_vehicles = sum(vehicle_counts.values())

    green_times = {}
    for phase, count in vehicle_counts.items():
        if total_vehicles == 0:
            green = available_green / len(vehicle_counts)
        else:
            green = max(min_green_time, (count / total_vehicles) * available_green)

        # Weather adjustment
        if weather_condition == "Rain/Storm":
            green += 5
        elif weather_condition == "Fog/Smoke":
            green *= 1.2

        green_times[phase] = min(int(green), 60)  # Cap to 60 seconds

    return green_times

# Simulation function
def simulate_traffic(green_times, emergency_phases, yellow_time):
    phase_pairs = [("Phase 1", "Phase 3"), ("Phase 2", "Phase 4")]
    
    # Emergency handling
    if emergency_phases:
        st.session_state.emergency_detected = True
        st.session_state.emergency_phases = emergency_phases
        
        with st.expander("🚨 Emergency Vehicle Detected", expanded=True):
            for phase in emergency_phases:
                st.write(f"✅ Giving GREEN priority to {phase} for 15 seconds")
                time.sleep(1)  # For demo purposes
    
    # Normal operation
    with st.expander("🚦 Normal Traffic Signal Cycle", expanded=True):
        for pair in phase_pairs:
            p1, p2 = pair
            green_time = max(green_times[p1], green_times[p2])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{p1} & {p2} GREEN Time", f"{green_time}s")
            with col2:
                st.metric("YELLOW Time", f"{yellow_time}s")
            with col3:
                red_time = sum(green_times[p] + yellow_time for p in green_times if p not in pair)
                st.metric("RED Time for Others", f"{red_time}s")
            
            # Simulate signal change
            with st.empty():
                st.success(f"🟢 GREEN ON: {p1} + {p2}")
                time.sleep(2)
                st.warning("🟡 YELLOW ON")
                time.sleep(1)
                st.error("🔴 RED ON")

# Main app logic
if st.button("Run Traffic Simulation"):
    if not (phase1_img and phase2_img and phase3_img and phase4_img):
        st.warning("Please upload all four phase images")
    else:
        with st.spinner("Processing images and detecting vehicles..."):
            vehicle_counts, emergency_phases = process_images()
        
        # Process weather data
        weather_data = {}
        if weather_json:
            weather_data = json.load(weather_json)
            condition = weather_data.get('weather', [{}])[0].get('description', 'clear').lower()
            temp = weather_data.get('main', {}).get('temp', 25)
            
            with st.expander("🌦️ Weather Information", expanded=True):
                st.write(f"Condition: {condition}")
                st.write(f"Temperature: {temp}°C")
        
        # Allocate green times
        green_times = allocate_green_times(
            vehicle_counts,
            weather_condition,
            total_cycle_time,
            yellow_time,
            min_green_time
        )
        
        # Display allocation
        with st.expander("📊 Green Time Allocation", expanded=True):
            for phase, time in green_times.items():
                st.write(f"{phase}: {time} seconds")
        
        # Run simulation
        simulate_traffic(green_times, emergency_phases, yellow_time)

# Instructions
st.sidebar.markdown("""
### Instructions:
1. Upload images for all 4 traffic phases
2. Optionally upload weather data (JSON)
3. Adjust cycle parameters as needed
4. Click "Run Traffic Simulation"
""")
