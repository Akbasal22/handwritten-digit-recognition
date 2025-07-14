import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

model = load_model("digit_model.h5")

def img_to_array(path):
    img = cv2.imread(path, 0)
    new_img = img/255
    final = new_img.reshape(1, -1)
    final = 1 - final
    return final

def test():
    final = img_to_array("./test.png")
    prediction = model.predict(final)
    print(f"Prediction is : {np.argmax(prediction)}")
    print(prediction)

def main():
    x = '1'
    while (x=='1'):
        test()
        x = input("do you want to run again? press 1 for yes, press 0 for no")
main()