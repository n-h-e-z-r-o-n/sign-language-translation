from keras.models import load_model  # install Keras
import numpy as np  # Install nampy for math
import tensorflow as tf
import cv2
from cvzone.HandTrackingModule import HandDetector
import math

detect_hand = HandDetector(maxHands=1)


model = load_model('my_model.h5', compile=False)  # Load the model_details
class_names = open('./Letters/labels.txt', 'r').readlines()  # Load the labels

imageSize = 240
img  = tf.keras.utils.load_img('test_imag_2.jpeg', target_size=(imageSize, imageSize))

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print('predictions:', predictions)
print('Class:', class_names[np.argmax(score)], end='')
