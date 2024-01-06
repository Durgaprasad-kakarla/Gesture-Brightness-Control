import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self,mode=False,maxHands=4,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon# It gives threshold value for hand detection
        self.trackCon = trackCon# It gives threshold value for hand tracking in the image
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) #Information about detecting hands and their landmarks
        if self.results.multi_hand_landmarks:#It gives landmarks for multiple hands
            for handLMS in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLMS.landmark):
                    if draw:
                        self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self,img,handNo=0,draw=True):
        self.lmList=[]
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                h,w,c=img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id,cx,cy])
        return self.lmList
    def fingersUp(self):
        fingers=[]
        tipIds=[4,8,12,16,20]
        if len(self.lmList) != 0:
            if self.lmList[tipIds[0]][1] < self.lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            for id in range(1, 5):
                if self.lmList[tipIds[id]][2] < self.lmList[tipIds[id] - 1][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=10):
        """
        Find the distance between two landmarks input should be (x1,y1) (x2,y2)
        :param p1: Point1 (x1,y1)
        :param p2: Point2 (x2,y2)
        :param img: Image to draw output on. If no image input output img is None
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """
        if self.lmList:
            p1=self.lmList[p1][1:]
            p2=self.lmList[p2][1:]
            x1, y1 = p1
            x2, y2 = p2
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            length = math.hypot(x2 - x1, y2 - y1)
            info = (x1, y1, x2, y2, cx, cy)
            if img is not None:
                cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
                cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
                cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

            return length
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector=handDetector()
    while True:
        success, img = cap.read()
        lmList=detector.findPosition(img)
        img=detector.findHands(img)
        detector.findDistance(8,12,img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__=='__main__':
    main()