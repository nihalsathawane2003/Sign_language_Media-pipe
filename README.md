# **Sign Language Detection and Speech Synthesis**

This project implements **American, Indian, and German Sign Language Detection** using **Mediapipe** and **Flask**. The system detects alphabets from sign language gestures, constructs meaningful words, and converts these words into speech using Text-to-Speech (TTS). It is designed to facilitate communication for individuals using sign language.

---

## **Features**
- **Multi-language support**: 
  - Detects alphabets from **American Sign Language (ASL)**, **Indian Sign Language (ISL)**, and **German Sign Language (GSL)**.
- **Alphabet Detection**: Uses **Mediapipe** for real-time gesture recognition.
- **Word Formation**: Automatically forms meaningful words from detected alphabets.
- **Speech Synthesis**: Speaks the detected words aloud using Js.
- **Web Application**: A simple and user-friendly interface built with **Flask**.

---

## **Technologies Used**
- **Mediapipe**: For real-time hand tracking and gesture detection.
- **Flask**: To create a web-based user interface for detection.
- **Python**: For training models, real-time detection, and speech synthesis.
- **Text-to-Speech (TTS)**: Converts the formed words into speech .
- **Kaggle Datasets**: Used for training models.

---

## **Project Workflow**
1. **Dataset Preparation**:
   - Download datasets for **ASL**, **ISL**, and **GSL** from Kaggle. (Links below)
   - Preprocess datasets and use them to train models for alphabet recognition.

2. **Model Training**:
   - Run the provided `.ipynb` files for each language:
     - `American_Training.ipynb`
     - `Indian_Training.ipynb`
     - `German_Training.ipynb`

3. **Integration with Flask**:
   - The trained models are integrated into a Flask application.
   - The web interface allows users to upload videos or use the camera for real-time sign language detection.

4. **Real-Time Detection**:
   - Detects alphabets, forms words, and speaks them aloud.

---

## **Setup Instructions**

### Prerequisites:
- Python 3.8 or later
- Necessary Python libraries:
  - `Flask`
  - `opencv-python`
  - `mediapipe`
  - `matplotlib` 
  - `scikit-learn`
  - `numpy` 

### Steps to Run the Project:
1. Clone this repository:
   ```bash
   git clone https://github.com/nihalsathawane2003/Sign_language_Media-pipe

