# pedestrians-emotional-state-detection-
This project focuses on detecting and classifying facial expression using HOG (Histogram of Oriented  Gradients) and SIFT (Scale-Invariant Feature Transform) for feature extraction, combined  with any classical machine learning classification algorithm (e.g., SVM, Random Forest,  Logistic Regression).

# Problem Statement 
Understanding a pedestrian’s emotional state can help autonomous vehicles make safer decisions.
For example:

Fear or surprise may indicate an unpredictable action (e.g., suddenly crossing the street).

Happiness or neutral may suggest calm and predictable movement.

# DataSet
CK+ (Extended Cohn-Kanade) dataset:
https://www.kaggle.com/datasets/shawon10/ckplus/data
It contains 593 video sequences from 123 subjects, with 7 labeled emotion categories.

# Steps:

# 1. Image preprocessing
Extract frames from video sequences

Convert to grayscale

Resize to 128×128 pixels

Normalize pixel values

(Optional) Apply histogram equalization

# 2.Feature Extraction
Tried 2 methods were tested:
- HOG(histogram of oriented gradient) with SVM (Support vctor classification model) for classification with accuracy of 0.98.
- SIFT for feature extraction and SVM fro classification with accuracy 0.842

# 3. Real-time emotion prediction
- Face detection using OpenCV’s Haar Cascade
(haarcascade_frontalface_default.xml)
- Preprocess detected faces (grayscale, resize, normalize)
- Extract features using HOG
- Load trained model (joblib.load('emotion_classifier.pkl'))
- Predict and overlay emotion label on video frames

  # Tools & Libraries used:
  Python 3.x
  OpenCV
  Scikit-learn
  NumPy
  Matplotlib
  
