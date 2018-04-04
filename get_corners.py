#!/usr/bin/python3.5
import cv2
import sys
import numpy

if len(sys.argv) < 2:
    print('You should run `./get_corners.py <image>`')
    sys.exit(-1)

# read the image from the given file
image_file = sys.argv[1]
frame = cv2.imread(image_file)

# if couldn't find the frame, stop the program
if frame is None:
    print('Frame is not found in the given file!')
    sys.exit(-1)

# convert the frame to gray
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# get harris corners of the image
corners = cv2.cornerHarris(gray, 2, 5, 0.04, cv2.BORDER_DEFAULT)
# dialate the corners so they seem bigger.
dst = cv2.dilate(corners, None)

# for dst points bigger than %4 of the max,
# mark the frame pixels red
frame[dst>0.04*dst.max()]=[0,0,255]

# show the result
cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
