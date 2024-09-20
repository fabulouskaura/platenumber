import cv2
import numpy as np
import pytesseract
import tkinter

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# To capture a plate number in a video streaming
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 80)
m = 0
while True:
    m = m + 1
    check, image = cap.read()
    key = cv2.waitKey()
    if key == ord('s'):
        break
showpic = cv2.imwrite('PlateNumber.jpg', image)

image = cv2.imread('PlateNumber.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
Filter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(Filter, 20, 180)
cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST,  cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break
if screenCnt is None:
    detected = 0
    print("No contour detected")
else:
    detected = 1
if detected == 1:
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv2.bitwise_and(image, image, mask=mask)
    # Now crop
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

    (thresh, blackAndWhiteImage) = cv2.threshold(Cropped, 180, 255, cv2.THRESH_BINARY)
    # Text Recognition
    text = pytesseract.image_to_string(Cropped, config='--psm 7')

    print("Detected Number is:", text.upper())
    cv2.imshow("Frame", image)
    cv2.imshow('Cropped', Cropped)
    cv2.imshow('Gray', gray)
    cv2.imshow('Edged', edged)

cv2.waitKey(0)

cv2.destroyAllWindows()
