#!/usr/bin/python3.5
import cv2
import sys
import numpy as np
import json

if len(sys.argv) < 2:
    print("Please call `./calibrate_camera_by_video.py <video>` to calibrate your camera.")
    sys.exit(-1)

# Start the video capture with the given video file
video_path = sys.argv[1]
video_capture = cv2.VideoCapture(video_path)
patternSize = (7, 5)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)

# create the arguments of the calibrate camera function.
objpoints = []
imgpoints = []
imgSize = None
frameCount = 0

# while the camera is open,
while video_capture.isOpened():
    ret, frame = video_capture.read()
    # stop if we can't read a frame from the video e.g. its finished
    if ret is False:
        break
    frameCount += 1
    if frameCount % 50 != 1:
        continue

    # convert the image to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # find the chessboard pattern with the given size
    ret, corners = cv2.findChessboardCorners(gray, patternSize)
    # store the current image's size to be used in calibrate camera.
    imgSize = gray.shape[::-1]

    # if we have found corners for the frame
    if ret is True and corners is not None:
        # refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # append the object points
        objpoints.append(objp)
        # append the image points at the same order with objpoints
        imgpoints.append(corners2)

# given a list of matching object points and image points, calculate the camera calibration arguments
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgSize, None, None)

print(json.dumps({
    'rms': rms,
    'camera_matrix': [[f for f in arr] for arr in camera_matrix],
    'dist_coefs': [f for f in dist_coefs[0]]
}))
