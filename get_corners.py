#!/usr/bin/python3.5
import cv2
import sys
import numpy

if len(sys.argv) < 2:
    print('You should run `./get_corners.py <image>`')
    sys.exit(-1)

image_file = sys.argv[1]
frameRaw = cv2.imread(image_file)
frame = cv2.resize(frameRaw, (1280, 720))

if frame is None:
    print('Frame is not found in the given file!')
    sys.exit(-1)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
corners = cv2.cornerHarris(gray, 2, 5, 0.04, cv2.BORDER_DEFAULT)
dst = cv2.dilate(corners, None)

frame[dst>0.04*dst.max()]=[0,0,255]

cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
