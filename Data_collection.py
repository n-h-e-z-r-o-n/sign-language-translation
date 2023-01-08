import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector

DataSet_folder = 'DataSet/words/you'  # current images we are collection
counter = 0  # keep track of number of images in each class

camera = cv2.VideoCapture(0)

detect_hand = HandDetector(maxHands=1)
offset = 20
imageSize = 240
while True:
    ret, image = camera.read()
    image_copy = image.copy()
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
            try:
              image_white[:, wGap: wCal + wGap] = imageReSize
            except:
                continue

        else:
            k = imageSize / w
            hCal = math.ceil(k * h)
            try:
                imageReSize = cv2.resize(imagCop, (imageSize, hCal))
            except:
                continue
            hGap = math.ceil((imageSize - hCal) / 2)
            try:
              image_white[hGap: hCal + hGap, :] = imageReSize
            except:
                continue

    cv2.imshow('coppy image Image', image_copy)
    try:
      cv2.imshow('image_white', image_white)
    except:
        pass

    key = cv2.waitKey(1)
    if key ==ord('s'):  # press s to save each image
        counter += 1
        cv2.imwrite(f'{DataSet_folder}/image_{counter}.jpg', image_white)
        print(counter)

camera.release()
cv2.destroyAllWindows()

