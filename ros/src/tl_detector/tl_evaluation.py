import os
import io
import glob
import cv2
import numpy as np
import matplotlib.image as mpimg

from light_classification.tl_classifier import TLClassifier

WRITE_FILE = '../../../wrongly_classified_image.csv'
DATA_DIR = '../../../tl_classifier_evaluation/tl_exhaustive_evaluation_data/'

filelist = glob.glob(DATA_DIR + "*")
light_score_th = 0.4
light_class_dict = {1:0, 2:2, 3:1}

light_classifier = TLClassifier()
f = open(WRITE_FILE, 'wb')
f.write('filename')
f.write(',')
f.write('true_state')
f.write(',')
f.write('predicted_state')
f.write('\n')
f.close()


for file in filelist:
    true_state = int(file.split('-')[-1].split('.')[0])
    img = mpimg.imread(file)
    img = np.expand_dims(img, axis=0)
    boxes, scores, classes, num = light_classifier.get_classification(img)
    if num > 0 and scores[0][0] > light_score_th:
        predicted_state = light_class_dict[classes[0][0]]
    else:
        predicted_state = 4  # unknown

    if true_state != predicted_state:
        with open(WRITE_FILE, 'a') as f:
            f.write(file.split('/')[-1])
            f.write(',')
            f.write(str(true_state))
            f.write(',')
            f.write(str(int(predicted_state)))
            f.write('\n')
