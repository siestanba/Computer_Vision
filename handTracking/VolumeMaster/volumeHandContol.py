# Rajout des fps
 
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import HandTrackingModule as htm
import os

hCam, wCam = 640, 480

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7) # Augmenter detectionCon permet de rendre la détectino plus naturelle (moins d'erreurs)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # 1 = inversion horizontale
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False) # Car on dessine déjà
    if len(lmList) != 0:
        print(lmList[4], lmList[8]) # On affiche le pouce et l'index (extrémités)

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2  

        cv2.circle(img, (x1, y1), 8, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 8, (255,0,255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255,0,255), 3)
        cv2.circle(img, (cx, cy), 8, (255,0,255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        #print(length) # Pour vérifier la range de length

        # Handrange: 50 -> 300
        # Volrange: 0 -> 100

        vol = np.interp(length, [50,300], [0,100])
        print(vol)

        cv2.rectangle(img, (50,400-2*int(vol)), (85,400), (255,0,0), cv2.FILLED)
        cv2.putText(img, f'{int(vol)}',(48,430), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)

        # 150 -> 100
        # 400 -> 0
        
        os.system(f'osascript -e "set volume output volume {vol}"') # Fonction pour modifier le son

        if length<50:
            cv2.circle(img, (cx, cy), 8, (0,255,0), cv2.FILLED)

    cv2.rectangle(img, (50,200), (85,400), (255,0,0), 3)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}',(10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
