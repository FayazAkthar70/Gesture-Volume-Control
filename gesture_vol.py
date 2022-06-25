import math
import cv2 as cv
import time
import hand_tracking.handTrackingModule as htm
from subprocess import call

vid = cv.VideoCapture(0)
current_time = 0
prev_time = 0
length = 0
prev_length = 0
detector = htm.handDetector()

while True:
    isTrue, img = vid.read()
    img = detector.find_hand(img)
    hands_list = detector.find_position(img)
    if hands_list:
        index_x, index_y = hands_list[8][1:]
        thumb_x, thumb_y = hands_list[4][1:]
        
        cx, cy = (thumb_x + index_x)//2,(thumb_y + index_y)//2
        cv.circle(img, (thumb_x,thumb_y), 6, (0, 255, 0), -1)
        cv.circle(img, (index_x,index_y), 6, (0, 255, 0), -1)
        cv.circle(img, (cx,cy), 6, (0, 255, 0), -1)
        cv.line(img, (thumb_x,thumb_y), (index_x,index_y), (0, 255, 0), 2)
        
        length = math.hypot(thumb_x - index_x, thumb_y - index_y)        
        if length>(prev_length*1.1):
            call(["amixer", "-D", "pulse", "sset", "Master", "10%+"])
            prev_length = length
        if length<(prev_length*0.9):
            call(["amixer", "-D", "pulse", "sset", "Master", "10%-"])
            prev_length = length

    current_time = time.time()
    fps = str(int(1/(current_time-prev_time)))
    prev_time = current_time
    cv.putText(img, fps, (10,70), cv.FONT_HERSHEY_COMPLEX, 2,(0,0,0), 3)
    img = cv.resize(img,(1100,800))
    cv.imshow('video',img)
    if cv.waitKey(10) & 0xFF == ord("q"):   #press q to close video
        break
    
vid.release()
cv.destroyAllWindows()