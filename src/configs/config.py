import cv2

#### ANNOTATION ON WEBCAM ####
text = {
    'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
    'org': (10, 450),  # bottom left corner of text
    'fontScale': 1,
    'color': (255, 255, 255),
    'lineType': 2
}

#### HAND POSITION RECOGNITION ####
confidence_treshold = 0.9
path_to_model = 'RPS_classifier.joblib'
classes = {0: 'rock', 1: 'paper', 2: 'scissors'}
