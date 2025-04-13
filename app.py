import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Initialize MediaPipe modules
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Landmark indices for facial features
left_eye_indices = [33, 160, 158, 133, 153, 144, 33]
right_eye_indices = [362, 385, 387, 263, 373, 380, 362]
nose_indices = [240, 460]
mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 61]

def process_frame(image):
    try:
        # Resize and flip image
        target_width = 640
        aspect_ratio = image.shape[1] / image.shape[0]
        target_height = int(target_width / aspect_ratio)
        image = cv2.resize(image, (target_width, target_height))
        image = cv2.flip(image, 1)  # Mirror image for webcam
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        face_results = face_mesh.process(image_rgb)
        hand_results = hands.process(image_rgb)
        annotated_image = image.copy()

        # Draw face landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                # Left eye
                for i in range(len(left_eye_indices) - 1):
                    start_idx = left_eye_indices[i]
                    end_idx = left_eye_indices[i + 1]
                    start_landmark = face_landmarks.landmark[start_idx]
                    end_landmark = face_landmarks.landmark[end_idx]
                    start_x = int(start_landmark.x * annotated_image.shape[1])
                    start_y = int(start_landmark.y * annotated_image.shape[0])
                    end_x = int(end_landmark.x * annotated_image.shape[1])
                    end_y = int(end_landmark.y * annotated_image.shape[0])
                    cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 1)
                # Right eye
                for i in range(len(right_eye_indices) - 1):
                    start_idx = right_eye_indices[i]
                    end_idx = right_eye_indices[i + 1]
                    start_landmark = face_landmarks.landmark[start_idx]
                    end_landmark = face_landmarks.landmark[end_idx]
                    start_x = int(start_landmark.x * annotated_image.shape[1])
                    start_y = int(start_landmark.y * annotated_image.shape[0])
                    end_x = int(end_landmark.x * annotated_image.shape[1])
                    end_y = int(end_landmark.y * annotated_image.shape[0])
                    cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 1)
                # Nose
                for idx in nose_indices:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * annotated_image.shape[1])
                    y = int(landmark.y * annotated_image.shape[0])
                    cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
                # Mouth
                for i in range(len(mouth_indices) - 1):
                    start_idx = mouth_indices[i]
                    end_idx = mouth_indices[i + 1]
                    start_landmark = face_landmarks.landmark[start_idx]
                    end_landmark = face_landmarks.landmark[end_idx]
                    start_x = int(start_landmark.x * annotated_image.shape[1])
                    start_y = int(start_landmark.y * annotated_image.shape[0])
                    end_x = int(end_landmark.x * annotated_image.shape[1])
                    end_y = int(end_landmark.y * annotated_image.shape[0])
                    cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )

        return annotated_image
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
        return image

st.title("LandMarkFinder")

# Image Upload Section
st.header("image section")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Failed to load image.")
        else:
            annotated_image = process_frame(image)
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_image_rgb, caption="Detected Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error processing uploaded image: {str(e)}")

# Camera Section
st.header("Webcam Detection")
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False
if "cap" not in st.session_state:
    st.session_state.cap = None

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Camera"):
        if not st.session_state.camera_active:
            st.session_state.cap = cv2.VideoCapture(0)
            if st.session_state.cap.isOpened():
                st.session_state.camera_active = True
            else:
                st.error("Error: Could not access camera.")
                st.session_state.cap = None
with col2:
    if st.button("Stop Camera"):
        st.session_state.camera_active = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
            st.empty()  # Clear frame

frame_placeholder = st.empty()
if st.session_state.camera_active and st.session_state.cap is not None:
    try:
        while st.session_state.camera_active:
            success, frame = st.session_state.cap.read()
            if not success:
                st.error("Failed to capture frame.")
                st.session_state.camera_active = False
                break
            annotated_frame = process_frame(frame)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_frame_rgb, caption="Camera Feed", use_column_width=True)
    except Exception as e:
        st.error(f"Error in camera feed: {str(e)}")
    finally:
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
            st.session_state.camera_active = False
            frame_placeholder.empty()

st.write("Face and hand LandMark Detector")