#!/usr/bin/python3.5
import sys
import cv2
import numpy as np

if len(sys.argv) < 2:
    print("Please call `./find_contours.py <image_file>`")
    sys.exit(-1)

image_path = sys.argv[1]
image = cv2.imread(image_path)
kernel = np.ones((5,5),np.uint8)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
median = cv2.medianBlur(gray, 3)
gradient = cv2.morphologyEx(median, cv2.MORPH_GRADIENT, kernel)
threshold = cv2.adaptiveThreshold(gradient, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
canny = cv2.Canny(gray, 150, 200, 3)

_, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def checkIsSquare(approx):
    return len(approx) == 4 and cv2.isContourConvex(approx)

squares = []
for contour in contours:
    approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.02, True)
    if checkIsSquare(approx) is False:
        continue

    area = np.fabs(cv2.contourArea(approx))
    squares.append(approx)

bigs = sorted(squares, key=cv2.contourArea, reverse=True)[0]
cv2.drawContours(image, [bigs], -1, (0, 0, 255), 3)

cv2.imshow('open', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
