import cv2
import mediapipe as mp
import numpy as np
import time
import math
import HandTrackingModule as htm
import pyautogui
from queue import Queue
import threading

class VideoStreamWidget:
    def __init__(self, src=1):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(3, 640)
        self.capture.set(4, 480)
        
        # Create a queue to store frames
        self.frame_queue = Queue(maxsize=2)
        self.stopped = False
        
        # Start frame capture thread
        self.thread = threading.Thread(target=self.update_frame, args=())
        self.thread.daemon = True
        self.thread.start()

    def update_frame(self):
        while not self.stopped:
            if not self.frame_queue.full():
                ret, frame = self.capture.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    if self.frame_queue.empty():
                        self.frame_queue.put(frame)
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage

    def get_frame(self):
        return self.frame_queue.get() if not self.frame_queue.empty() else None

    def stop(self):
        self.stopped = True
        self.capture.release()
        self.thread.join()

class MouseController:
    def __init__(self):
        pyautogui.FAILSAFE = False
        self.screenWidth, self.screenHeight = pyautogui.size()
        self.smoothening = 2
        self.prev_x, self.prev_y = 0, 0
        self.command_queue = Queue(maxsize=10)
        self.stopped = False
        
        # Start mouse control thread
        self.thread = threading.Thread(target=self.process_commands)
        self.thread.daemon = True
        self.thread.start()

    def move_mouse(self, x, y):
        # Convert coordinates
        x3 = np.interp(x, [100, 540], [0, self.screenWidth])
        y3 = np.interp(y, [100, 380], [0, self.screenHeight])
        
        # Smoothen values
        curr_x = self.prev_x + (x3 - self.prev_x) / self.smoothening
        curr_y = self.prev_y + (y3 - self.prev_y) / self.smoothening
        
        self.command_queue.put(('move', (curr_x, curr_y)))
        self.prev_x, self.prev_y = curr_x, curr_y

    def click(self):
        self.command_queue.put(('click', None))

    def right_click(self):
        self.command_queue.put(('right_click', None))

    def process_commands(self):
        while not self.stopped:
            if not self.command_queue.empty():
                command, args = self.command_queue.get()
                if command == 'move':
                    pyautogui.moveTo(*args)
                elif command == 'click':
                    pyautogui.click()
                    time.sleep(0.3)
                elif command == 'right_click':
                    pyautogui.rightClick()
                    time.sleep(0.3)
            time.sleep(0.01)

    def stop(self):
        self.stopped = True
        self.thread.join()

class VirtualMouse:
    def __init__(self):
        self.video_stream = VideoStreamWidget()
        self.mouse_controller = MouseController()
        self.detector = htm.handDetector(detectionCon=0.7, maxHands=1)
        self.pTime = 0
        self.stopped = False

    def process_hand(self, img):
        img = self.detector.findHands(img)
        lmList = self.detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            # Index and thumb points
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            # Check which fingers are up
            fingers = self.detector.fingersUp()

            # Moving Mode - Index finger up
            if fingers[1] == 1 and fingers[2] == 0:
                self.mouse_controller.move_mouse(x2, y2)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

            # Clicking Mode - Index and Thumb
            if fingers[1] == 1 and fingers[0] == 1:
                length = math.hypot(x2-x1, y2-y1)
                if length < 40:
                    cv2.circle(img, ((x1+x2)//2, (y1+y2)//2), 15, (0, 255, 0), cv2.FILLED)
                    self.mouse_controller.click()

            # Right Click Mode - Index and Middle finger
            if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0:
                length = math.hypot(lmList[8][1]-lmList[12][1], lmList[8][2]-lmList[12][2])
                if length < 40:
                    cv2.circle(img, ((lmList[8][1]+lmList[12][1])//2, 
                                   (lmList[8][2]+lmList[12][2])//2), 15, (0, 0, 255), cv2.FILLED)
                    self.mouse_controller.right_click()

        return img

    def run(self):
        while not self.stopped:
            frame = self.video_stream.get_frame()
            if frame is not None:
                # Process hand and mouse control
                img = self.process_hand(frame)

                # Calculate and display FPS
                cTime = time.time()
                fps = 1/(cTime-self.pTime)
                self.pTime = cTime
                cv2.putText(img, f'FPS: {int(fps)}', (20, 50), 
                           cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

                # Show image
                cv2.imshow("Virtual Mouse", img)

            # Break loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

    def stop(self):
        self.stopped = True
        self.video_stream.stop()
        self.mouse_controller.stop()
        cv2.destroyAllWindows()

def main():
    virtual_mouse = VirtualMouse()
    try:
        virtual_mouse.run()
    except KeyboardInterrupt:
        virtual_mouse.stop()

if __name__ == "__main__":
    main()
