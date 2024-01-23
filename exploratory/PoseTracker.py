import cv2
import mediapipe as mp


class PoseTracker:
    def __init__(self, mode=False, complexity=1, smooth=True, enable_seg=False, smooth_seg=False, detectTreshold=0.5, trackingTreshold=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.detectTreshold = 0.5
        self.trackingTreshold = 0.5
        self.current_result = None

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.detector = self.mp_pose.Pose(self.mode, self.complexity, self.smooth, self.enable_seg, self.smooth_seg, self.detectTreshold, self.trackingTreshold)

    def find_pose(self, img, dic_output=True):
        self.current_result = self.detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h, w, _ = img.shape
        poses = {}
        if self.current_result.pose_landmarks and dic_output:
            for index, landmark in enumerate(self.current_result.pose_landmarks):
                poses[index] = (landmark.x*w, landmark.y*h, landmark.z)
            return poses

    def draw_pose(self, img):
        self.find_pose(img, dic_output=False)
        if self.current_result.pose_landmarks:
            self.mp_drawing.draw_landmarks(img, self.current_result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
