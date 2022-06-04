import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model
import tensorflow as tf
import numpy as np
from collections import deque



classes_list = ["Punch", "YoYo", "Swing", "HorseRace"]

st.title("Video Classification")
uploaded_video = st.file_uploader("Choose video", type=["mp4","mov"])

writer = None
frame_skip = 10

model = load_model('model_Loss_0.7431485056877136_Accuracy_0.7622950673103333.h5')
if uploaded_video is not None: # run only when user uploads video
    vid = uploaded_video.name
    with open(vid, mode='wb') as f:
        f.write(uploaded_video.read()) # save video to disk

    st.markdown(f"""
    ### Files
    - {vid}
    """,
    unsafe_allow_html=True) # display file name

    vidcap = cv2.VideoCapture(vid) # load video from disk


def predict_video(video_file_path, output_file_path, window_size):
    
    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen = window_size)

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)

    # Getting the width and height of the video 
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'x264'), 24, (original_video_width, original_video_height))

    while True: 

        # Reading The Frame
        status, frame = video_reader.read() 

        if not status:
            break

        resized_frame = cv2.resize(frame, (64, 64))    
        normalized_frame = resized_frame / 255
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        if len(predicted_labels_probabilities_deque) == window_size:
            
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)

            # Calculating Average of Predicted Labels Probabilities Column Wise 
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)

            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list[predicted_label]
          
            # Overlaying Class Name Text Ontop of the Frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        video_writer.write(frame)

        video_reader.release()
    video_writer.release()
# Setting sthe Window Size which will be used by the Rolling Average Proces
window_size = 1

output_video_file_path = 'result.mp4'
if uploaded_video is not None:
    predict_video(uploaded_video.name, output_video_file_path, window_size)
    st.header("The output video")
    st.video('result.mp4')

    