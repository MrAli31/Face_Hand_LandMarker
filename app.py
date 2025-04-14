import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Detect if running on Streamlit Cloud
def is_running_on_streamlit_cloud():
    try:
        import streamlit.runtime.scriptrunner.script_run_context as context
        ctx = context.get_script_run_context()
        return ctx is not None and "streamlit" in ctx.session_id
    except:
        return False

is_cloud = is_running_on_streamlit_cloud()

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
        target_width = 640
        aspect_ratio = image.shape[1] / image.shape[0]
        target_height = int(target_width / aspect_ratio)
        image = cv2.resize(image, (target_width, target_height))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_results = face_mesh.process(image_rgb)
        hand_results = hands.process(image_rgb)
        annotated_image = image.copy()

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                # Draw eye, nose, mouth
                for feature_indices, color in zip(
                    [left_eye_indices, right_eye_indices, mouth_indices],
                    [(0, 0, 255), (0, 0, 255), (255, 0, 0)]
                ):
                    for i in range(len(feature_indices) - 1):
                        start = face_landmarks.landmark[feature_indices[i]]
                        end = face_landmarks.landmark[feature_indices[i + 1]]
                        sx, sy = int(start.x * annotated_image.shape[1]), int(start.y * annotated_image.shape[0])
                        ex, ey = int(end.x * annotated_image.shape[1]), int(end.y * annotated_image.shape[0])
                        cv2.line(annotated_image, (sx, sy), (ex, ey), color, 1)

                for idx in nose_indices:
                    point = face_landmarks.landmark[idx]
                    x, y = int(point.x * annotated_image.shape[1]), int(point.y * annotated_image.shape[0])
                    cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

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
        st.error(f"Error processing image: {str(e)}")
        return image

st.title("LandMarkFinder")
st.write("Face and Hand Landmark Detector")

# Image Upload Section
st.header("Image Upload")
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
            st.image(annotated_image_rgb, caption="Detected Landmarks", use_column_width=True)
    except Exception as e:
        st.error(f"Error processing uploaded image: {str(e)}")

# Webcam Section (local only)
if not is_cloud:
    st.header("Webcam Detection")
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False
    if "cap" not in st.session_state:
        st.session_state.cap = None

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Camera"):
            if not st.session_state.camera_active:
                for index in [0, 1, 2, 3]:
                    st.session_state.cap = cv2.VideoCapture(index)
                    if st.session_state.cap.isOpened():
                        st.session_state.camera_active = True
                        break
                if not st.session_state.camera_active:
                    st.error("Error: Could not access camera on indices 0–3. Check connection or permissions.")
                    st.session_state.cap = None
    with col2:
        if st.button("Stop Camera"):
            st.session_state.camera_active = False
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
                st.empty()

    frame_placeholder = st.empty()
    if st.session_state.camera_active and st.session_state.cap is not None:
        try:
            while st.session_state.camera_active:
                success, frame = st.session_state.cap.read()
                if not success:
                    st.error("Failed to capture frame.")
                    st.session_state.camera_active = False
                    break
                frame = cv2.flip(frame, 1)
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
else:
    st.info("🚫 Webcam detection is not supported in Streamlit Cloud.\nPlease use the image upload section above.")
