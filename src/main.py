import cv2
from HandTracker import HandTracker
import configs.config as config
import joblib
import numpy as np

# not safe at all but ignore complaints about not having named features in predict
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

def main():
    cap = cv2.VideoCapture(0)
    hand_tracker = HandTracker(maxHands=1, detectTreshold=0.7)
    model = joblib.load(config.path_to_model)
    confidence_treshold = config.confidence_treshold
    index_to_classes = config.classes

    while 1:
        success, image = cap.read()
        if success:
            hands = hand_tracker.find_hands(image, keep_normalized=True)
            image = cv2.flip(image, 1)
            if hands:
                X = [x for y in hands[0] for x in y]
                X = np.array(X)
                probas = model.predict_proba(X.reshape(1,-1))
                if np.amax(probas) > confidence_treshold:
                    cv2.putText(image, index_to_classes[(np.argmax(probas)+1)%len(index_to_classes)], **config.text)



        cv2.imshow('Webcam', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

if __name__ == '__main__':
    main()
