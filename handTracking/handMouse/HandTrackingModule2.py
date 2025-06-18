# Création du module pour utiliser dans d'autres projets
 
import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = float(detectionCon) 
        self.trackCon = float(trackCon)

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        
        # Points de repère des bouts des doigts
        self.tipIds = [4, 8, 12, 16, 20]  # [pouce, index, majeur, annulaire, auriculaire]
        self.lmList = []  # Initialisation de lmList comme attribut de classe
        self.results = None

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                             self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results and self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > handNo:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self, lmList):
        fingers = []
        if len(lmList) == 0:
            return [0, 0, 0, 0, 0]

        # Thumb
        if lmList[self.tipIds[0]][1] < lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findAllHands(self, img, draw=True):
        hands = []
        if self.results.multi_hand_landmarks:
            for idx, (handLms, handedness) in enumerate(zip(self.results.multi_hand_landmarks, 
                                                          self.results.multi_handedness)):
                myHand = {}
                mylmList = []
                
                # Obtenir tous les points de repère
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    mylmList.append([id, cx, cy])
                
                # Déterminer si c'est la main droite ou gauche
                myHand["type"] = handedness.classification[0].label
                myHand["lmList"] = mylmList
                
                hands.append(myHand)
                
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                
        return hands


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    

    detector = handDetector()

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

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()