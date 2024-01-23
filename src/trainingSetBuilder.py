import pandas as pd
import cv2
import numpy as np
from HandTracker import HandTracker
import configs.config as display_config
from time import time
import yaml

def main(path_to_dataset_config='./configs/dataset_builder_config.yml'):
    with open(path_to_dataset_config, 'r') as f:
        config = yaml.safe_load(f)
    cap = cv2.VideoCapture(0)
    hand_tracker = HandTracker(maxHands=1, detectTreshold=0.7)
    dataset = []
    sequence = config['sequence']
    n_class = len(sequence)
    class_to_oh = {}
    for i in range(n_class):
        base = [0]*n_class
        base[i] = 1
        class_to_oh[list(sequence.keys())[i]] = base
    print(class_to_oh)

    while 1:
        for r in range(config['n_repeat']):
            for classe, duration in sequence.items():
                start = time()
                while time() - start < config['transition_time']:
                    success, image = cap.read()
                    if success:
                        image = cv2.flip(image, 1)
                        cv2.putText(image, classe + ' in ' + str(np.round(config['transition_time'] - (time() - start), 2)) + ' s',
                                    **display_config.text)
                        cv2.imshow('Webcam', image)
                        if cv2.waitKey(5) & 0xFF == 27:
                            break
                start = time()
                while time() - start < duration:
                    success, image = cap.read()
                    if success:
                        hands = hand_tracker.find_hands(image, keep_normalized=True)
                        if hands:
                            hand_tracker.draw_hands(image)
                            dataset.append([x for y in hands[0] for x in y] + class_to_oh[classe])
                        image = cv2.flip(image, 1)
                        cv2.putText(image, classe, **display_config.text)
                        cv2.imshow('Webcam', image)
                        if cv2.waitKey(5) & 0xFF == 27:
                            break

        columns = [str(y)+x for y in range(21) for x in ('X', 'Y', 'Z')] + list(class_to_oh.keys())
        training_set = pd.DataFrame(dataset, columns=columns)
        training_set.to_csv(config['out_name'], index=False)
        break

if __name__ == '__main__':
    main()
