import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLOv8 model
model_path = "best.pt"  # Change this if needed
model = YOLO(model_path)

st.title("Classroom Attendance Tracker 🏫")

# Sidebar for navigation
option = st.sidebar.radio("Choose an option", ["Upload Image", "Live Camera"])

# Function to process images
def detect_people_in_image(image):
    frame = np.array(image)  # Convert PIL image to numpy array
    results = model(frame)  # Run YOLO detection
    detected_objects = results[0].boxes

    num_students = 0  # Count detected people
    for box in detected_objects:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())

        if cls == 0:  # Class 0 is 'person' in COCO dataset
            num_students += 1
            label = f'Person {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, num_students

# Upload Image Option
if option == "Upload Image":
    st.subheader("Upload an Image for Attendance Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)  # Open image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process Image
        processed_image, count = detect_people_in_image(image)

        # Show results
        st.image(processed_image, caption=f"Detected Attendance: {count}", use_column_width=True)
        st.write(f"### Attendance Count: {count} 🧑‍🎓")

# Live Camera Option
elif option == "Live Camera":
    st.subheader("Live Attendance Tracking 🎥")
    st.write("Click 'Start Camera' to begin tracking.")

    # Start button
    start = st.button("Start Camera")

    if start:
        cap = cv2.VideoCapture(0)  # Open webcam
        frame_placeholder = st.empty()  # Placeholder for updating frames
        attendance_text = st.empty()  # Placeholder for updating attendance count

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame.")
                break

            results = model(frame)  # Run YOLO detection
            detected_objects = results[0].boxes

            num_students = 0  # Count detected people
            for box in detected_objects:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                if cls == 0:
                    num_students += 1
                    label = f'Person {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert frame to RGB (OpenCV uses BGR, but Streamlit requires RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update the Streamlit image placeholder
            frame_placeholder.image(frame, caption="Live Attendance Tracking", use_column_width=True)
            attendance_text.write(f"### Current Attendance Count: {num_students} 🧑‍🎓")

        cap.release()
