import cv2 as cv
import time
import sys

from tensorflow.keras.models import load_model
model = load_model(sys.argv[1] + '.h5')
# model = load_model("CNN.model")
class_name = ["closed_left", "closed_right", "open_left", "open_right" ]
cam = cv.VideoCapture(0)
while True:
    start = time.time()
    _, img = cam.read()
    cv.imshow('my webcam', img)

    # if cv.waitKey(1) == 119:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, (24, 24))
    gray = gray.reshape(1, 24, 24, 1)
    y_pred = model.predict(gray)
    end = time.time()
    print("Execution time: " + str(end-start))
    print(y_pred)
    print(class_name[list(y_pred[0]).index(max(y_pred[0]))])

    if cv.waitKey(1) == 27:
        break


cam.release()
cv.destroyAllWindows()
