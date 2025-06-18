import cv2
import mediapipe as mp
import time
import sys

print(sys.version)

cap = cv2.VideoCapture(1)


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB) #c'est une fonction dans mediapie qui traite l'image et nous renvoie le résultat
    print(results.multi_hand_landmarks)




    cv2.imshow("Image", img)
    cv2.waitKey(1)
