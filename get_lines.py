#!/usr/bin/python3.5
import cv2
import sys
import numpy
import utils

# read the image from the file
if len(sys.argv) < 2:
    print("You should run `./get_squares.py <image>`")
    sys.exit(-1)

image_file = sys.argv[1]
frameRaw = cv2.imread(image_file)
# resize to landscape hd
frame = cv2.resize(frameRaw, (1280, 720))

# exit if no frame has been found
if frame is None:
    print("Frame is not found in the given file.")
    sys.exit(-1)

# set the color scheme to gray
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# find canny edges with hardcoded tresholds.
edges = cv2.Canny(gray, 150, 200, 3)
# calculate houghlines over the canny thresholds
lines = cv2.HoughLines(edges, 1, numpy.pi / 180, 170)

# for each line in the result
for lineArr in lines:
    for line in lineArr:
        rho, theta = line
        # draw the line by converting polar representations to pair of points
        (p1, p2) = utils.convertRhoThetaToCoordinates(rho, theta)
        cv2.line(frame, p1, p2, (0, 255, 0), 2)

cv2.imshow("Resulting frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
