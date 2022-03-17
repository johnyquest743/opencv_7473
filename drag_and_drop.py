'''
Installed python 3.7, opencv, cvzone packages
'''

import cv2
import cvzone
import mediapipe as mp
import time
from cvzone.HandTrackingModule import HandDetector
import math

fpsReader = cvzone.FPS()

frame_x_dim = 1280
frame_y_dim = 720

vid_cap = cv2.VideoCapture(1)
vid_cap.set(3, frame_x_dim)
vid_cap.set(4, frame_y_dim)
detector = HandDetector(detectionCon=0.6)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 2, 1, 0.5, 0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

cx = 200
cy = 100
w = 200
h = 200

f_color = (0, 0, 0)
rect_color = (225, 0, 255)
new_rect_color = (255, 255, 0)


def find_dist_cent(pt1, pt2):
    ## Function to find distance between two coordinates
    x1, y1 = pt1.x, pt1.y
    x2, y2 = pt2.x, pt2.y

    dist = math.hypot(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return dist, (cx, cy)


while vid_cap.isOpened():
    success, img = vid_cap.read()
    img = cv2.flip(img, 1)  ## If using rear camera uncomment this line

    if success:
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            hand1 = None
            hand2 = None

            if len(results.multi_hand_landmarks) > 0:
                hand1 = results.multi_hand_landmarks[0]
                hand2 = results.multi_hand_landmarks[1] if len(results.multi_hand_landmarks) > 1 else None

                dist_p1_p2 = find_dist_cent(hand1.landmark[8], hand1.landmark[12])

                # print(cx - w // 2, cx + w // 2, hand1.landmark[8].x, hand1.landmark[8].x * 1280)
                # print(cy - h // 2, cy + h // 2, hand1.landmark[8].y, hand1.landmark[8].y * 720)

                if (cx - w // 2 < hand1.landmark[8].x * frame_x_dim < cx + w // 2) & (cy - h // 2 < hand1.landmark[8].y * frame_y_dim < cy + h // 2):
                    f_color = new_rect_color
                    if dist_p1_p2[0] < 0.10:
                        cx, cy = math.ceil( hand1.landmark[8].x * frame_x_dim ), math.ceil( hand1.landmark[8].y * frame_y_dim )
                        print('^^^'*20)
                        print("Fin1 and Fin2 merged - ", dist_p1_p2)
                        print(cx, ' --- ', cy)
                        print('^^^'*20)
                    elif dist_p1_p2[0] >= 0.10:
                        f_color = rect_color

                        # cx, cy = math.ceil(hand1.landmark[8].x * 1280), math.ceil(hand1.landmark[8].y * 720)
                else:
                    f_color = rect_color

            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        rectangle1 = cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), f_color, cv2.FILLED)
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("My WEBCAM", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            vid_cap.release()
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
    else:
        break

vid_cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
