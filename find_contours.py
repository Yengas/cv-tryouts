#!/usr/bin/python3.5
import sys
import cv2
import numpy as np

if len(sys.argv) < 2:
    print("Please call `./find_contours.py <image_file>`")
    sys.exit(-1)

image_path = sys.argv[1]
image = cv2.imread(image_path)
# create a kernel to be useed by the gradient operation.
kernel = np.ones((5,5),np.uint8)

# make the image gray for easier processing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# get the median blur of the image to enhance edge detection
median = cv2.medianBlur(gray, 3)
# apply gradient morphology to make the strong edges more complete
gradient = cv2.morphologyEx(median, cv2.MORPH_GRADIENT, kernel)
# adaptive threshold to make canny edge work better.
threshold = cv2.adaptiveThreshold(gradient, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# get the canny edges of the image
canny = cv2.Canny(gray, 150, 200, 3)

# find contours on the canny edges by creating tree hierarchy, simple approx of points
_, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# a function to check the contour has 4 corners, and is a convex
def checkIsSquare(approx):
    return len(approx) == 4 and cv2.isContourConvex(approx)

squares = []
for contour in contours:
    # get the contour perimiter with poly approximation
    approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.02, True)
    # if not square, pass
    if checkIsSquare(approx) is False:
        continue

    #area = np.fabs(cv2.contourArea(approx)) # maybe check the area?
    # append the squares
    squares.append(approx)

# sort the squares by area, and get the biggest
bigs = sorted(squares, key=cv2.contourArea, reverse=True)[0]
# draw the biggest square to the original image.
cv2.drawContours(image, [bigs], -1, (0, 0, 255), 3)

# show the modified original image
cv2.imshow('open', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
