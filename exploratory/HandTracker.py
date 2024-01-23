import cv2
import mediapipe as mp
from scipy.spatial import distance


class HandTracker:
    def __init__(self, mode=False, maxHands=2, detectTreshold=0.5, trackingTreshold=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectTreshold = 0.5
        self.trackingTreshold = 0.5
        self.current_result = None

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(self.mode, self.maxHands, self.detectTreshold, self.trackingTreshold)

    def find_hands(self, img, dic_output=True, keep_normalized=False):
        self.current_result = self.detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h, w, _ = img.shape
        hands = {}
        if self.current_result.multi_hand_landmarks and dic_output:
            for n_hand, landmarks in enumerate(self.current_result.multi_hand_landmarks):
                for index, landmark in enumerate(landmarks.landmark):
                    if n_hand in hands.keys():
                        if keep_normalized:
                            hands[n_hand].append((landmark.x, landmark.y, landmark.z))
                        else:
                            hands[n_hand].append((int(landmark.x*w), int(landmark.y*h), landmark.z))
                    else:
                        hands[n_hand] = []
                        if keep_normalized:
                            hands[n_hand].append((landmark.x, landmark.y, landmark.z))
                        else:
                            hands[n_hand].append((int(landmark.x * w), int(landmark.y * h), landmark.z))
            return hands

    def draw_hands(self, img):
        self.find_hands(img, dic_output=False)
        if self.current_result.multi_hand_landmarks:
            for landmark in self.current_result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, landmark, self.mp_hands.HAND_CONNECTIONS)

    def is_middle_finger(self, img, test=False):
        finger_pos = self.find_hands(img)
        if finger_pos:
            finger_pos = finger_pos[0]
            middle_finger = distance.euclidean((finger_pos[12][0], finger_pos[12][1]),(finger_pos[9][0], finger_pos[9][1]))
            r_index = distance.euclidean((finger_pos[8][0], finger_pos[8][1]),(finger_pos[5][0], finger_pos[5][1]))/middle_finger
            r_ring_finger = distance.euclidean((finger_pos[16][0], finger_pos[16][1]),(finger_pos[13][0], finger_pos[13][1]))/middle_finger
            r_pinky = distance.euclidean((finger_pos[20][0], finger_pos[20][1]),(finger_pos[17][0], finger_pos[17][1]))/middle_finger

            if r_pinky < 0.5 and r_index < 0.5 and r_ring_finger <0.5:
                if test:
                    cv2.circle(img, (finger_pos[12][0], finger_pos[12][1]), 15, (255,0,0), -1)
                return True, finger_pos

        return False, None

    def is_fist(self, img, test=False):
        finger_pos = self.find_hands(img)
        if finger_pos:
            finger_pos = finger_pos[0]
            hand_size = distance.euclidean((finger_pos[0][0], finger_pos[0][1]),(finger_pos[5][0], finger_pos[5][1]))
            r_thumb = distance.euclidean((finger_pos[4][0], finger_pos[4][1]),(finger_pos[9][0], finger_pos[9][1]))/hand_size
            r_middle_finger = distance.euclidean((finger_pos[12][0], finger_pos[12][1]),(finger_pos[9][0], finger_pos[9][1]))/hand_size
            r_index = distance.euclidean((finger_pos[8][0], finger_pos[8][1]),(finger_pos[5][0], finger_pos[5][1]))/hand_size
            r_ring_finger = distance.euclidean((finger_pos[16][0], finger_pos[16][1]),(finger_pos[13][0], finger_pos[13][1]))/hand_size
            r_pinky = distance.euclidean((finger_pos[20][0], finger_pos[20][1]),(finger_pos[17][0], finger_pos[17][1]))/hand_size

            if r_pinky < 0.5 and r_index < 0.5 and r_ring_finger <0.5 and r_thumb < 0.5 and r_middle_finger < 0.5:
                if test:
                    cv2.circle(img, (finger_pos[0][0], finger_pos[0][1]), 15, (255,0,0), -1)
                return True, finger_pos

        return False, None



