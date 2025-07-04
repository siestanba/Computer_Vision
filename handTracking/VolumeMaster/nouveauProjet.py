# Projet de test pour importer le module

import cv2
import mediapipe as mp
import time

import HandTrackingModule as htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(1) # Choix de la sortie vidéo


detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # 1 = inversion horizontale
    if not success:
        continue # si arrive pas à lire on skip 

    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[8])


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_COMPLEX, 3,
                (255,0,255), 3)

    cv2.imshow("Say hi!", img)
    cv2.waitKey(1)