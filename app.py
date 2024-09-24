import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from mtcnn import MTCNN
from tempfile import NamedTemporaryFile

# Load the trained model
model = load_model(r'D:\ML\DeepFake_Detection\deepfake_detection_model.h5')

# MTCNN for face detection
detector = MTCNN()

# Function to detect and preprocess face from an image
def extract_and_preprocess_face(img):
    faces = detector.detect_faces(img)
    if len(faces) == 0:
        return None
    # Extract the bounding box of the first face detected
    x, y, width, height = faces[0]['box']
    face = img[y:y+height, x:x+width]
    # Resize face to 224x224 pixels
    face_resized = cv2.resize(face, (224, 224))
    face_array = image.img_to_array(face_resized)
    face_array = np.expand_dims(face_array, axis=0)
    face_array = preprocess_input(face_array)
    return face_array

# Function to predict the probability of being fake
def predict_fake_probability(img):
    face_array = extract_and_preprocess_face(img)
    if face_array is None:
        return 0.5  # Default to 0.5 if no face is detected
    prediction = model.predict(face_array)
    return prediction[0][0]

# Streamlit app
st.title('Deepfake Detector: AI-Driven Real-Time Image Authentication')

# Upload image or video
st.sidebar.title('Upload an Image')
uploaded_file = st.sidebar.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

# If an image is uploaded
if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        # Display the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_file = cv2.imdecode(file_bytes, 1)
        st.image(image_file, channels="BGR", caption="Uploaded Image", use_column_width=True)

        # Predict real or fake
        if st.button('Predict'):
            fake_prob = predict_fake_probability(image_file)
            if fake_prob >= 0.5:
                st.write(f"The image is FAKE with a confidence of {fake_prob:.2f}.")
            else:
                st.write(f"The image is REAL with a confidence of {1 - fake_prob:.2f}.")

    # # If a video is uploaded
    # elif uploaded_file.type.startswith('video'):
    #     temp_file = NamedTemporaryFile(delete=False)
    #     temp_file.write(uploaded_file.read())
        
    #     st.video(temp_file.name)

    #     if st.button('Predict Video'):
    #         video_capture = cv2.VideoCapture(temp_file.name)
    #         frame_count = 0
    #         fake_scores = []

    #         while True:
    #             ret, frame = video_capture.read()
    #             if not ret:
    #                 break
                
    #             frame_count += 1
    #             if frame_count % 10 == 0:
    #                 fake_prob = predict_fake_probability(frame)
    #                 fake_scores.append(fake_prob)

    #         video_capture.release()

    #         if len(fake_scores) > 0:
    #             avg_fake_score = np.mean(fake_scores)
    #             if avg_fake_score >= 0.5:
    #                 st.write(f"The video is FAKE with a confidence of {avg_fake_score:.2f}.")
    #             else:
    #                 st.write(f"The video is REAL with a confidence of {1 - avg_fake_score:.2f}.")
    #         else:
    #             st.write("No face detected in the video.")
