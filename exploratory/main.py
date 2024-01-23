import cv2
import HandTracker
from effectDrawer import ImageEffectDrawer


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    hand_tracker = HandTracker.HandTracker(detectTreshold=0.7)
    effect_fist = ImageEffectDrawer('../ressources/rock.jpg', mask_treshold=(245,245,245))
    effect_md = ImageEffectDrawer('../ressources/brick.jpg')

    while 1:
        success, image = cap.read()
        if success:
            is_middle_finger_flag, finger_pos = hand_tracker.is_middle_finger(image)
            is_fist_flag, hand_pos = hand_tracker.is_fist(image)
            if is_middle_finger_flag:
                effect_md.draw_on_middle_finger(image, finger_pos)
            elif is_fist_flag:
                effect_fist.draw_on_fist(image, hand_pos)
            #hand_tracker.draw_hands(image)

        cv2.imshow('Webcam', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
