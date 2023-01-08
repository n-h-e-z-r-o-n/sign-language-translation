import math
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector


def model_f(hand_image):
    img_array = tf.keras.utils.img_to_array(hand_image)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    sign = labels[np.argmax(score)]
    return f'{sign[3:]}  {int(100 * np.max(score))}%'


model = load_model('models/words_model/words_model.h5', compile=False)  # Load the model_details
labels = open('models/words_model/words_labels.txt', 'r').readlines()  # Load the labels

camera = cv2.VideoCapture(0)
detect_hand = HandDetector(maxHands=2)
offset = 20
imageSize = 240
while True:
    ret, image = camera.read()  # Capture frame-by-frame
    image_copy = image.copy()  # Make copy of the Capture frame
    hands, frame_hand = detect_hand.findHands(image)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        image_white = np.ones((imageSize, imageSize, 3), np.uint8) * 255
        imagCop = image[y - offset:y + h + offset, x - offset:x + w + offset]
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imageSize / h
            wCal = math.ceil(k * w)
            try:
                imageReSize = cv2.resize(imagCop, (wCal, imageSize))
            except:
                continue
            wGap = math.ceil((imageSize - wCal) / 2)
            image_white[:, wGap: wCal + wGap] = imageReSize

            prediction = model_f(cv2.resize(image_white, (240, 240), interpolation=cv2.INTER_AREA))
        else:
            k = imageSize / w
            hCal = math.ceil(k * h)
            try:
                imageReSize = cv2.resize(imagCop, (imageSize, hCal))
            except:
                continue
            hGap = math.ceil((imageSize - hCal) / 2)
            image_white[hGap: hCal + hGap, :] = imageReSize
            prediction = model_f(cv2.resize(image_white, (240, 240), interpolation=cv2.INTER_AREA))

        cv2.putText(image_copy, prediction, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

    cv2.imshow('Main Window', image_copy)
    try:
        cv2.imshow('Hand Sign', image_white)
    except:
        pass

    # Wait for the user to press a key
    if cv2.waitKey(1) == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
