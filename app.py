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
SEQUENCE_LENGTH = 20
model = load_model('model_Loss_0.15672941505908966_Accuracy_0.9637681245803833.h5')


def predict_video(video_file_path, output_file_path, SEQUENCE_LENGTH):
    
    # Reading the Video File using the VideoCapture Object

    video_reader = cv2.VideoCapture(video_file_path)

    # Getting the width and height of the video 
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))
    frames_queue = deque(maxlen = SEQUENCE_LENGTH)

    predicted_class_name = ''

    while video_reader.isOpened(): 

        # Reading The Frame
        ok, frame = video_reader.read() 

        if not ok:
            break

        resized_frame = cv2.resize(frame, (64, 64))
    
        normalized_frame = resized_frame / 255

        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            

            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis = 0))[0]

            # Get the index of class with highest probability.
            predicted_label = np.argmax(predicted_labels_probabilities)

            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list[predicted_label]
          
        # Write predicted class name on top of the frame.
        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        video_writer.write(frame)

    video_reader.release()
    video_writer.release()
    return predicted_class_name


def main():
    st.title("Video Classification")
    uploaded_video = st.file_uploader("Choose video", type=["mp4","mov"])


    if uploaded_video is not None: # run only when user uploads video

        vid = uploaded_video.name
        vidcap = cv2.VideoCapture(vid) # load video from disk

        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read()) # save video to disk
    if st.button('Classifier la video'):    
        output_video_file_path = 'result.mp4'
        if uploaded_video is not None:
            rs = predict_video(uploaded_video.name, output_video_file_path, SEQUENCE_LENGTH)
            st.success('La classe de la video est {}'.format(rs))
            st.header("The output video")
            st.video('result.mp4')

if __name__ == '__main__':
    main()

    
