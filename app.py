import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Load the YOLOv8 model
model = YOLO('best.pt')  # Replace 'best.pt' with the path to your trained model

# Streamlit app title
st.title("Plastic Pollution Detection App")

# Upload an image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    image_np = np.array(image)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Perform inference
    st.write("Detecting objects...")
    results = model.predict(source=image_cv2, save=False)  # Perform prediction

    # Annotate the image
    annotated_image = results[0].plot()  # Get annotated image
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  # Convert back to RGB

    # Display annotated image
    st.image(annotated_image, caption="Detected Objects", use_column_width=True)

    # Show detected objects with confidence scores
    st.write("Detected Objects:")
    for box in results[0].boxes:
        st.write(f"{results[0].names[int(box.cls[0])]}: {box.conf[0]:.2f}")

