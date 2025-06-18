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
    vol = os.popen('osascript -e "output volume of (get volume settings)"').read() # On récupère le son de l'ordinatuer


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

        # Variable statique pour stocker la position initiale et le volume initial
        if not hasattr(detector, 'initial_cy'):
            detector.initial_cy = None
            detector.initial_vol = None

        if length<50:
            if detector.initial_cy is None:
                # Capture la position initiale et le volume actuel au moment du pinch
                detector.initial_cy = cy
                # Récupérer le volume actuel
                current_vol = os.popen('osascript -e "output volume of (get volume settings)"').read()
                detector.initial_vol = float(current_vol)
            
            # Calculer le changement relatif
            delta_y = detector.initial_cy - cy
            vol = detector.initial_vol + (delta_y / 2)  # Diviser par 2 pour rendre le contrôle moins sensible
            vol = np.clip(vol, 0, 100)  # S'assurer que le volume reste entre 0 et 100
            
            cv2.circle(img, (cx, cy), 8, (0,255,0), cv2.FILLED)
            cv2.rectangle(img, (50,400-2*int(vol)), (85,400), (255,0,0), cv2.FILLED)
            cv2.putText(img, f'{int(vol)}',(48,430), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
            
            os.system(f'osascript -e "set volume output volume {vol}"')
        else:
            # Réinitialiser les valeurs initiales quand on relâche le pinch
            detector.initial_cy = None
            detector.initial_vol = None

    cv2.rectangle(img, (50,200), (85,400), (255,0,0), 3)
    cv2.rectangle(img, (50,400-2*int(vol)), (85,400), (255,0,0), cv2.FILLED)
    cv2.putText(img, f'{int(vol)}',(48,430), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}',(10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
