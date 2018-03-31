#!/usr/bin/python3.5
import cv2
import sys
import numpy

if len(sys.argv) < 2:
    print("You should run `./get_squares.py <image>`")
    sys.exit(-1)

image_file = sys.argv[1]

def showCorners(patternSize, flag):
    frame = cv2.imread(image_file)

    if frame is None:
        print("Frame is not found in the given file.")
        sys.exit(-1)

    found, corners = cv2.findChessboardCorners(frame, patternSize, flag)
    if found is False:
       return
    cv2.drawChessboardCorners(frame, patternSize, corners, found)

    cv2.imshow('Size: %dx%d' % patternSize, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for i in range(3, 9):
    for j in range(3, 9):
        showCorners((i, j), cv2.CALIB_CB_FAST_CHECK)
