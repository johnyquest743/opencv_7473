'''
Installed python 3.7, opencv, cvzone, mediapipe, pyttsx3, pyautogui packages
'''

import cv2
import cvzone
import mediapipe as mp
import time
from cvzone.HandTrackingModule import HandDetector
import math
import subprocess
import pyttsx3
import numpy as np
import vlc
import pyautogui


## we are using Numpy.linalg.norm function to compute euclidean distance between two points

### LANDMARKS FOR FACE, LIPS, LEFT EYE, RIGHT EYES --
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
       185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]

RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]

LEFT_EYE_TOP_BOTTOM = [386, 374]
LEFT_EYE_LEFT_RIGHT = [263, 362]

RIGHT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_LEFT_RIGHT = [133, 33]

UPPER_LOWER_LIPS = [13, 14]
LEFT_RIGHT_LIPS = [78, 308]

FACE=[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
       377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

frame_x_dim = 1280
frame_y_dim = 720

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(1)

def eye_blink(img, landmarks, side='left'):
    # print(landmarks)
    arr2pro = LEFT_EYE_TOP_BOTTOM if side == 'left' else RIGHT_EYE_TOP_BOTTOM
    eye_top_bottom = []

    for m in arr2pro:
        x, y = math.ceil(landmarks.landmark[m].x * frame_x_dim), math.ceil(landmarks.landmark[m].y * frame_y_dim)
        eye_top_bottom.append(np.array([x, y]))

    dist = np.linalg.norm(eye_top_bottom[0] - eye_top_bottom[1])
    return dist


with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Error .....")
            continue

        img_w, img_h = image.shape[0:2]

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        speech = pyttsx3.init()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                le_blink = eye_blink(image, face_landmarks, 'left')
                re_blink = eye_blink(image, face_landmarks, 'right')

                ## Using Left Eye Blink to open a url in the browser
                ## Adjusting the distance between landmarks of left eye  as per your requirement - sensitivity
                if le_blink < 5:
                    ## Using speech object to notify user for alert
                    # speech.say('Hello Avyukt and Dolly ...')
                    # speech.runAndWait()

                    # creating vlc media player object and open a media file in vlc media player
                    # media = vlc.MediaPlayer("path to media file.mp4")
                    # media.play()

                    import webbrowser
                    webbrowser.open("www.google.com")

                ## Using Right Eye Blink to close a tab in the browser using key combinations with pytautogui package
                ## Adjusting the distance between landmarks of right eye  as per your requirement - sensitivity
                if re_blink < 7:
                    pyautogui.hotkey('ctrl', 'w')

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_IRISES,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()