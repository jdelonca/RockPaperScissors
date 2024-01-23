import cv2
import numpy as np
from scipy.spatial import distance


class VideoEffectDrawer:
    def __init__(self, path_to_video_effect):
        cap = cv2.VideoCapture(path_to_video_effect)
        self.frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.effect = np.empty((self.frameCount, self.frameHeight, self.frameWidth, 3), np.dtype('uint8'))
        self.current_frame_id = -1

        self.pos = 0
        self.wanted_size = (0, 0)

        fc = 0
        ret = True
        while (fc < self.frameCount and ret):
            ret, self.effect[fc] = cap.read()
            fc += 1
        cap.release()

    def draw_first_frame(self, img, size, pos):
        self.pos = pos
        self.size = size
        self.current_frame_id = 0

    def draw_next_frame(self, img):
        if self.current_frame_id >= 0:

            self.current_frame_id += 1
            if self.current_frame_id > self.frameCount:
                self.current_frame_id = -1
        else:
            pass


class ImageEffectDrawer:
    def __init__(self, path_to_image, mask_treshold=None):
        self.effect = cv2.imread(path_to_image)
        if mask_treshold:
            self.mask = np.full(self.effect.shape[0:2], True)
            for i, treshold_per_channel in enumerate(mask_treshold):
                self.mask = np.logical_and(self.mask, self.effect[:, :, i] > treshold_per_channel)
        else:
            self.mask = np.full(self.effect.shape[0:2], False)
        self.mask = np.dstack([self.mask.astype(float)]*self.effect.shape[2])

    def draw_effect(self, img, size, pos):
        h, w, c = img.shape
        x_min, y_min = pos
        x_max = x_min + size[0] if x_min + size[0] < w else w
        y_max = y_min + size[1] if y_min + size[1] < h else h
        resized_effect = cv2.resize(self.effect, (x_max-x_min, y_max-y_min))
        resized_mask = cv2.resize(self.mask, (x_max-x_min, y_max-y_min))
        img[y_min:y_max, x_min:x_max] = img[y_min:y_max, x_min:x_max]*resized_mask + resized_effect*(1-resized_mask)

    def draw_on_middle_finger(self, img, finger_pos):
        size_x = int(
            distance.euclidean((finger_pos[5][0], finger_pos[5][1]), (finger_pos[13][0], finger_pos[13][1])) * 0.75)
        size_y = int(distance.euclidean((finger_pos[12][0], finger_pos[12][1]), (finger_pos[9][0], finger_pos[9][1])))
        pos_x = finger_pos[12][0] - size_x // 2
        pos_y = finger_pos[12][1] - 10
        self.draw_effect(img, (size_x, size_y), (pos_x, pos_y))

    def draw_on_fist(self, img, finger_pos):
        size_x = abs(finger_pos[5][0] - finger_pos[17][0])
        size_y = abs(finger_pos[5][1] - finger_pos[0][1])
        pos_x = finger_pos[5][0]
        pos_y = finger_pos[5][1]
        self.draw_effect(img, (size_x, size_y), (pos_x, pos_y))