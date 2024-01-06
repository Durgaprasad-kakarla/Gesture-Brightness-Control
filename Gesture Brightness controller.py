import cv2
import mediapipe as mp
import time
import math
import pycaw
import numpy as np
import HandtrackingModule as htm
import screen_brightness_control as sbc
cap=cv2.VideoCapture(0)
pTime=0
brightB=400
brightPer=0
tracker=htm.handDetector(detectionCon=0.9)
while True:
    success,img=cap.read()
    img=cv2.flip(img,1)
    img=cv2.resize(img,(1100,700))
    tracker.findHands(img)
    lmList=tracker.findPosition(img,draw=True)
    if len(lmList)>=9:
        print(lmList[4],lmList[8])#thumb and forefinger landmarks
        thumb=lmList[4]
        forefinger=lmList[8]
        x1,y1=thumb[1],thumb[2]
        x2,y2=forefinger[1],forefinger[2]
        cx,cy=(thumb[1]+forefinger[1])//2,(thumb[2]+forefinger[2])//2
        cv2.circle(img,(thumb[1],thumb[2]),10,(0,0,255),cv2.FILLED)
        cv2.circle(img,(forefinger[1],forefinger[2]),10,(0,0,255),cv2.FILLED)
        cv2.line(img,(thumb[1],thumb[2]),(forefinger[1],forefinger[2]),(0,0,255),3)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
        length=math.hypot(x2-x1,y2-y1)
        print(length)
        print(sbc.get_brightness())
        bright=np.interp(length,[40,200],[0,100])
        brightB = np.interp(length, [40, 200], [400, 150])
        brightPer = np.interp(length, [40, 200], [0, 100])
        sbc.set_brightness(bright,display=0)
        if length<50:
            cv2.circle(img,(cx,cy),10,(255,255,255),cv2.FILLED)
    cv2.rectangle(img, (50, 150), (95, 400), (0, 255, 0), 3)
    if int(brightPer)==100:
        cv2.rectangle(img, (50, int(brightB)), (95, 400), (0,0,255), cv2.FILLED)
    else:
        cv2.rectangle(img, (50, int(brightB)), (95, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(int(brightPer)) + "%", (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,"FPS "+str(int(fps)),(20,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)
    cv2.imshow("Gesture Brightness Control",img)
    cv2.waitKey(1)