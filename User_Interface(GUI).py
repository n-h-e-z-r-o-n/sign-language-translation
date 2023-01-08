import cv2
import tkinter as tk
from PIL import Image, ImageTk
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import tensorflow as tf
from keras.models import load_model  # install Keras

# ================================== Global Variables ===================================================================================================
model_track = 0
model = None
labels = None

#  model_details and lables
word_m = 'model_details/models/words_model/words_model.h5'
word_l = 'model_details/models/words_model/words_labels.txt'
letter_m = 'model_details/models/letters_model/letters_model.h5'
letter_l = 'model_details/models/letters_model/letter_labels.txt'
numbers_m = 'model_details/models/numbers_model/numbers_model.h5'
numbers_l = 'model_details/models/numbers_model/numbers_labels.txt'


# ================================== Functions ==========================================================================================================

def change_model():
    global model_track
    global model
    global labels
    if model_track == 0:
        speaker_bt.config(text='word')
        model_path = word_m
        model_lables = word_l
        model_track = 1
    elif model_track == 1:
        speaker_bt.config(text='letters')
        model_path = letter_m
        model_lables = letter_l
        model_track = 2
    elif model_track == 2:
        speaker_bt.config(text='numbers')
        model_path = numbers_m
        model_lables = numbers_l
        model_track = 0

    model = load_model(model_path, compile=False)  # Load the model_details
    labels = open(model_lables, 'r').readlines()  # Load the labels


def model_f(hand_image):
    global model
    global labels
    img_array = tf.keras.utils.img_to_array(hand_image)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    sign = labels[np.argmax(score)]
    sign.replace(" ", "")
    out_s = f'{sign[3:]} '
    s_accuracy = f'{int(100 * np.max(score))}%'
    return out_s, s_accuracy


# =================================================== User Interface ================================================================================
root = tk.Tk()
root.title("")

root.minsize(1000, 950)
background = 'white'
content_frame = tk.Frame(root, bg=background)
content_frame.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

v_Lable = tk.Label(content_frame, bg=background)
v_Lable.place(relheight=0.4, relwidth=0.22, rely=0.02, relx=0.01)

v1_Lable = tk.Label(content_frame, bg=background)
v1_Lable.place(relheight=0.75, relwidth=0.75, rely=0.02, relx=0.24)

v2_Lable = tk.Label(content_frame, fg='red', font=('Courier New', 14, 'bold'), bg=background, anchor='w')
v2_Lable.place(relheight=0.05, relwidth=0.22, rely=0.43, relx=0.01)

tk.Label(content_frame, bg=background, text='Sign Translation :', font=('Courier New', 14, 'bold'), anchor='e').place(relx=0.01, rely=0.8, relwidth=0.2, relheight=0.05)
show_translation = tk.Label(content_frame, bg='white', fg='Brown', text='', font=('Courier New', 13, 'bold'), anchor='w')
show_translation.place(relx=0.22, rely=0.8, relwidth=0.6, relheight=0.05)

tk.Label(content_frame, bg=background, text='Accuracy :', font=('Courier New', 14, 'bold'), anchor='e').place(relx=0.01, rely=0.86, relwidth=0.2, relheight=0.05)
pu = tk.Label(content_frame, bg='white', fg='Brown', font=('Courier New', 13, 'bold'), anchor='w')
pu.place(relx=0.22, rely=0.86, relwidth=0.6, relheight=0.05)

tk.Label(content_frame, bg=background, text='Model change :', font=('Courier New', 14, 'bold'), anchor='e').place(relx=0.01, rely=0.92, relwidth=0.2, relheight=0.05)
speaker_bt = tk.Button(content_frame, bg='gray', fg='white', borderwidth=0, border=0, font=('Courier New', 14, 'bold'), command=change_model)
speaker_bt.place(relx=0.22, rely=0.92, relwidth=0.15, relheight=0.05)

change_model()

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 520)
detect_hand = HandDetector(maxHands=1)
offset = 20
imageSize = 224


def show_frame():
    _, frame = cap.read()
    frameCopy = frame.copy()
    hands = detect_hand.findHands(frame, draw=False)
    global Language_tl
    global C0MP
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        image_white = np.ones((imageSize, imageSize, 3), np.uint8) * 255
        imagCop = frame[y - offset:y + h + offset, x - offset:x + w + offset]
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imageSize / h
            wCal = math.ceil(k * w)
            try:
                imageReSize = cv2.resize(imagCop, (wCal, imageSize))
                v2_Lable.config(text='')
            except:
                v2_Lable.config(text='ERROR: Hand to close to camera')
                show_frame()
            wGap = math.ceil((imageSize - wCal) / 2)
            try:
                image_white[:, wGap: wCal + wGap] = imageReSize
            except:
                show_frame()
            sign, acuu = model_f(cv2.resize(image_white, (240, 240), interpolation=cv2.INTER_AREA))
        else:
            k = imageSize / w
            hCal = math.ceil(k * h)
            try:
                imageReSize = cv2.resize(imagCop, (imageSize, hCal))
                v2_Lable.config(text='')
            except:
                v2_Lable.config(text='ERROR: Hand to close to camera')
                show_frame()
            hGap = math.ceil((imageSize - hCal) / 2)
            try:
                image_white[hGap: hCal + hGap, :] = imageReSize
            except:
                show_frame()
            sign, acuu = model_f(cv2.resize(image_white, (240, 240), interpolation=cv2.INTER_AREA))

        cv2.putText(frameCopy, sign+acuu , (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        show_translation.config(text=sign)
        pu.config(text=acuu)

    try:
        photo = ImageTk.PhotoImage(image=Image.fromarray(image_white))
        v_Lable.imgtk = photo
        v_Lable.config(image=photo)
    except:
        v_Lable.config(text='waiting For Your Hand')

    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_BGR2RGBA)
    photo1 = ImageTk.PhotoImage(image=Image.fromarray(frameCopy))
    v1_Lable.imgtk = photo1
    v1_Lable.config(image=photo1)
    v_Lable.after(10, show_frame)


show_frame()
root.mainloop()
