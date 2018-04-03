#!/usr/bin/python3.5
import cv2
import sys
import utils
import numpy as np
from math import degrees

if len(sys.argv) < 2:
    print('You should run `./get_corners.py <image>`')
    sys.exit(-1)

def calculateAndShowLines(title, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 150, 200, 3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 170)

    # for each line in the result
    for lineArr in lines:
        for line in lineArr:
            rho, theta = line
            # draw the line by converting polar representations to pair of points
            p1, p2 = utils.convertRhoThetaToCoordinates(rho, theta)
            cv2.line(image, p1, p2, (0, 255, 0), 2)

            cv2.imshow(title + (' RHO: %f, Theta(degrees): %f' % (rho, degrees(theta))), image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# read the image from the given file
mtx, dist, _ = utils.readCalibration('./data/camera-calibration.json')
image_file = sys.argv[1]
image = cv2.imread(image_file)
h, w = image.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
distorted = cv2.undistort(image, mtx, dist, None, newCameraMatrix)

calculateAndShowLines('Original', image)
calculateAndShowLines('Distorted', distorted)
