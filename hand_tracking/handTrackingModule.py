import cv2 as cv
import mediapipe as mp
import time



class handDetector():
    def __init__(self, mode= False, max_hands=2, complexity=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detection_con = detection_confidence
        self.tracking_con = tracking_confidence
        
        self.mpHands = mp.solutions.mediapipe.python.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.complexity, self.detection_con, self.tracking_con)
        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
    
    def find_hand(self, img):
        imgRGB = cv.cvtColor(img, code=cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, hand, self.mpHands.HAND_CONNECTIONS)
        return img
            
    def find_position(self, img, draw=True, hand_no=0):
        hand_landmarks = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, landmark in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(w*landmark.x), int(h*landmark.y)
                hand_landmarks.append([id,cx,cy])
        return hand_landmarks
    

def main():
    current_time = 0
    prev_time = 0
    detector = handDetector()
    vid = cv.VideoCapture(0)
    
    while True:
        isTrue, img = vid.read()
        img = detector.find_hand(img)
        hand_landmarks = detector.find_position(img)
            
        current_time = time.time()
        fps = str(int(1/(current_time-prev_time)))
        prev_time = current_time
        cv.putText(img, fps, (10,70), cv.FONT_HERSHEY_COMPLEX, 2,(0,0,0), 3)
        
        cv.imshow('video',img)
        if cv.waitKey(10) & 0xFF == ord("q"):
            break
        
    vid.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
