import cv2
import HandTracker
import PoseTracker as PoseTracker
from scipy.spatial import distance
from subprocess import call
from time import sleep

max_length =300
def change_volume(volume):
    call(["amixer", "-D", "pulse", "sset", "Master", str(min(int(volume*100), 100))+"%"])
    sleep(0.05)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    hand_tracker = HandTracker.HandTracker(detectTreshold=0.7)
    pose_tracker = PoseTracker.PoseTracker()

    while 1:
        success, image = cap.read()
        if success:
            hands = hand_tracker.find_hands(image)
            #hand_tracker.draw_hands(image)
            if hands:
                x1, y1, x2, y2 = hands[0][4][0], hands[0][4][1], hands[0][8][0], hands[0][8][1]
                cv2.circle(image, (x1, y1), 15, cv2.FILLED)
                cv2.circle(image, (x2, y2), 15, cv2.FILLED)
                cv2.line(image, (x1, y1), (x2, y2), (255,255,0))
                length = distance.euclidean((x1, y1), (x2, y2))
                max_length = max(length, max_length)
                change_volume(length/max_length)

        cv2.imshow('Webcam', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
