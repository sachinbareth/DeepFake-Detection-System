# Deepfake Detection System

This document provides a step-by-step guide for building a deepfake detection system using Python, TensorFlow, Keras, MTCNN, and Streamlit for the web interface. 

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Step 1: Environment Setup](#step-1-environment-setup)
- [Step 2: Install Required Libraries](#step-2-Install-Required-Libraries)
- [Step 3: Building Web App with Streamlit](#step-3-Building-Web-App-with-Streamlit)

---

## Project Overview
The goal of this project is to build an AI-driven system that detects deepfake images by leveraging deep learning models, specifically a fine-tuned VGG16 model. We also use MTCNN for face detection and Flask or Streamlit to create a user-friendly web interface for image upload and analysis.

## Prerequisites
Before starting, ensure you have the following installed:
- Python 3.x
- TensorFlow/Keras
- OpenCV
- MTCNN
- Streamlit (or Flask if preferred)
- AWS Account for deployment
- Kaggle API for DFDC dataset

## Step 1: Environment Setup
1. **Install Python** (if not installed).
2. **Create Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate      # For Windows

## Step 2: Install Required Libraries

    pip install -r requirements.txt

## Step 3: Building Web App with Streamlit

    pip install streamlit

Run the App:

    streamlit run app.py



