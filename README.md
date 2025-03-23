# AI Workshop: Human Face Recognition & Attendance Counting

Welcome to the AI Workshop! 🎉 This repository contains all the necessary materials to train a YOLO model for human face recognition and deploy it on a Streamlit web application.

## Prerequisites ✅
Before diving in, ensure you have:
- **Python** (Version 3.8 & above)
- **Visual Studio Code** (VS Code)

## Software Tools 🛠️
We will be using the following platforms:
- [Roboflow](https://roboflow.com/) (for dataset preparation)
- [Google Colab](https://colab.research.google.com/) (for model training)
- [Streamlit](https://streamlit.io/) (for web deployment)

---
## Step 1: Dataset Preparation 📂
We will use a publicly available human face dataset from Roboflow.

### Dataset Links:
- [Reference 1](https://app.roboflow.com/logo-bplam/test-dxiix-ckery/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
- [Reference 2](https://universe.roboflow.com/logo-bplam/test-dxiix-ckery/dataset/1)

### Download Options:
1. **Using API (Recommended)**:
   - Click **"Show download code"** and then **"Continue"**.
   - Copy and paste the generated code into your Google Colab notebook.
   - Replace the `api_key` with your unique key.
2. **Manual Download**:
   - Click **"Download Dataset"** and save the zip file locally.
   - Upload the zip file to Google Colab.

---
## Step 2: Train YOLO Model 🏋️‍♂️
Refer to the following [Google Colab Notebook](https://colab.research.google.com/drive/18_2264gPhFC5G8KX6H3Z8PKNJW04K_M1?usp=sharing) for training.

### Training Steps:
1. Copy the Roboflow dataset API key and paste it into your Colab notebook.
2. Expand the file directory to verify dataset import.
3. Install dependencies using `pip install ultralytics`.
4. Set the correct dataset path and train the YOLO model for **20 epochs**.
5. Monitor training logs (this takes ~30 minutes, so grab a coffee ☕).
6. Check the mAP score (above **0.7** is considered good).
7. Visualize model performance using **F1 Curve, PR Curve, P Curve, and R Curve**.
8. Save the trained model and export the PyTorch weights for later use.

---
## Step 3: Implement YOLO Model on Streamlit 🖥️
### Steps:
1. Install required libraries using `pip install streamlit torch opencv-python`.
2. Create a `main.py` file and import necessary dependencies.
3. Load the trained YOLO model.
4. Implement an image processing function to allow image uploads for detection.
5. Implement real-time face recognition using a webcam.

---
## Step 4: Deploy on Streamlit 🚀
1. Run `streamlit run main.py` in the terminal.
2. A local Streamlit web app should open in your browser.

### Features:
✅ Upload an image to detect the number of attendees.
✅ Use a live camera to track attendance in real time.

---
## Final Output 🎯
✅ Recognizes human faces & counts attendance accurately!  
Hope you enjoy the workshop. Have a great day! 🎉  
**Don't forget to mark your attendance!** ✅

