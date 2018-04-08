#!/usr/bin/python3.5
import sys
import argparse
import cv2
import pickle
import numpy as np
from sklearn.externals import joblib
from find_chess_board_with_ml import findChessBoardCorners

parser = argparse.ArgumentParser(
    description='Given chessboard found from an image, and a video, tracks the chessboard in the video.'
)
parser.add_argument('board_placement', help='board placement object with found chessboard and frames')
parser.add_argument('video_path', help='video file to track the chessboard in')
args = parser.parse_args()

board_placement = pickle.load(open(args.board_placement, 'rb'))
video = cv2.VideoCapture(args.video_path)
corner_classifier = joblib.load('./data/corner-classifier-rf.pkl')

if video.isOpened() is False:
    print('Could not open the given video file for reading.')
    sys.exit(-1)

#orb = cv2.ORB_create()
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()
while video.isOpened():
    ret, frame = video.read()
    if ret is False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    source = cv2.cvtColor(board_placement.frame, cv2.COLOR_BGR2GRAY)

    #frame_kp, frame_desc = orb.detectAndCompute(gray, None)
    #source_kp, source_desc = orb.detectAndCompute(source, None)
    #frame_kp, frame_desc = sift.detectAndCompute(gray, None)
    #source_kp, source_desc = sift.detectAndCompute(source, None)
    frame_kp, frame_desc = findChessBoardCorners(corner_classifier, gray, frame)
    source_kp, source_desc = findChessBoardCorners(corner_classifier, source, board_placement.frame)

    matches = bf.knnMatch(frame_desc, source_desc, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) > 0:
        result = cv2.drawMatches(frame, frame_kp, board_placement.frame, source_kp, good, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow('result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
