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

