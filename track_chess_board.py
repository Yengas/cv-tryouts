#!/usr/bin/python3.5
import sys
import argparse
import cv2
from math import atan2, cos, sin
import numpy as np
from find_chess_board import findChessboardPlacement

# this class is based on the code of Nghia Ho who first coded it for video files
# and the code of chen jia who added kalman filter for live features.
# see http://nghiaho.com/uploads/videostabKalman.cpp
# python rewrite by yengas
class VideoStab:
    def __init__(self, prev, size, features_options = (200, 0.01, 30), prev_gray=None):
        # set the first frame as the previous and convert the color to gray
        self.prev = prev.copy()
        self.prev_gray = prev_gray.copy() if prev_gray is not None else cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        # set the size for the input and output image as a width/height tuple
        self.size = size
        # set the options to use when finding features
        self.features_options = features_options

        # x, y, and angle values accumulated by frames
        self.x, self.y, self.a = 0, 0, 0

        # posteriori state, priori estimate, posteriori estimate error covarience, priori estimate error covariance
        # gain and actual measuremeents
        self.X, self.X_, self.P, self.P_, self.K = [None for _ in range(0, 5)]
        # process and measurement noise covariance
        self.Q, self.R = np.full(3, 4e-3, dtype=np.double), np.full(3, 0.25, dtype=np.double)
        # keeping the last transformation just in case.
        self.last_T = np.zeros((2, 3), dtype=np.double)

    def process(self, cur, cur_gray=None):
        # set the first frame as the previous and convert the color to gray
        cur = cur.copy()
        cur_gray = cur_gray.copy() if cur_gray is not None else cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        # options to use
        (maxCorners, qualityLevel, minDistance) = self.features_options
        # calculate the features for the previous image
        prev_corners = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners, qualityLevel, minDistance)
        # find the current corners given the previous image and the current image
        cur_corners, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, cur_gray, prev_corners, None)
        # filter the corners by the optical flow status
        prev_corners, cur_corners = [
            np.array([x for i, x in enumerate(arr) if status[i] == 1]) for arr in [prev_corners, cur_corners]
        ]

        if len(prev_corners) == 0 or len(cur_corners) == 0:
            return prev
        # rigid transform no scaling/shearing
        T = cv2.estimateRigidTransform(prev_corners, cur_corners, False)
        T = T if T is not None else self.last_T

        # decompose t to variables
        dx, dy, da = T[0, 2], T[1, 2], atan2(T[1, 0], T[0, 0])
        # accumulated frame to frame transforms
        self.x, self.y, self.a = self.x + dx, self.y + dy, self.a + da
        z = np.array([self.x, self.y, self.a])

        if self.X is None:
            self.X = np.zeros(3, dtype=np.double)
            self.P = np.ones(3, dtype=np.double)
        else:
            # time update (prediction)
            self.X_ = self.X
            self.P_ = self.P + self.Q
            # measurement update (correction)
            self.K = self.P_ / (self.P + self.R)
            self.X = self.X + self.K * (z - self.X_)
            self.P = (np.ones(3, dtype=np.double) - self.K) * self.P_

        diff_x, diff_y, diff_a = self.X[0] - self.x, self.X[1] - self.y, self.X[2] - self.a
        dx, dy, da = dx + diff_x, dy + diff_y, da + diff_a
        T = np.array([
            [ cos(da), -sin(da), dx ],
            [ sin(da), cos(da), dy]
        ], dtype=np.double)

        self.prev, self.prev_gray = cur, cur_gray
        self.last_T = T
        return cv2.warpAffine(self.prev, T, self.size)

class FeatureTracking:
    def __init__(self, prev, size, features_options = (200, 0.01, 30), prev_gray=None):
        # set the first frame as the previous and convert the color to gray
        self.prev = prev.copy()
        self.prev_gray = prev_gray.copy() if prev_gray is not None else cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        # set the size for the input and output image as a width/height tuple
        self.size = size

        # setup sift finder and matcher
        self.sift = cv2.xfeatures2d.SIFT_create()
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.prev_kps, self.prev_descs = self.sift.detectAndCompute(self.prev_gray, None)


    def process(self, cur, cur_gray=None):
        # set the first frame as the previous and convert the color to gray
        cur = cur.copy()
        cur_gray = cur_gray.copy() if cur_gray is not None else cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        cur_kps, cur_descs = self.sift.detectAndCompute(cur_gray, None)
        # TODO: try changing between prev and cur
        matches = self.flann.knnMatch(cur_descs, self.prev_descs, k=2)

        # ratio test as per Lowe's paper
        matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

        if len(matches) > 300:
            src_pts = np.float32([ cur_kps[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ self.prev_kps[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

            homography, _ = cv2.findHomography(src_pts, dst_pts)
            #homography = np.ones((3, 3), dtype=np.double)
        else:
            return None

        self.prev, self.prev_gray = cur, cur_gray
        self.prev_kps, self.prev_descs = cur_kps, cur_descs
        return homography


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Given a video where the camera is static as possible, removes shakes and finds the chessboard'
    )
    parser.add_argument('video_path', help='video file to track the chessboard in')
    args = parser.parse_args()

    video = cv2.VideoCapture(args.video_path)# change to (0) for camera
    width, height, fps = [round(video.get(p)) for p in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS]]

    if video.isOpened() is False:
        print('Could not open the given video file for reading.')
        sys.exit(-1)

    ret, prev = video.read()
    if ret is False:
        print('Could not read the first frame.')
        sys.exit(-1)

    #videostab = VideoStab(prev, (width, height))
    tracking = FeatureTracking(prev, (width, height))
    pattern_size, board_size = (7, 7), (640, 640)
    last_homography, last_corners = None, None
    last_camera_homography = np.ones((3, 3), dtype=np.double)
    frame = 0

    while video.isOpened():
        ret, cur = video.read()
        if ret is False:
            break
        #result = videostab.process(cur)
        camera_homography = tracking.process(cur)
        if camera_homography is not None:
            last_camera_homography = camera_homography
        result = cur

        if last_homography is None and frame % 20 == 0:
            found, homography, corners = findChessboardPlacement(result, pattern_size, board_size, None, cv2.CALIB_CB_FAST_CHECK)

            if found:
                last_homography = homography
                last_corners = corners

        if last_homography is not None:
            board = cv2.warpPerspective(result, last_homography * last_camera_homography, board_size)
            cv2.imshow('board', board)

        cv2.imshow('camera', result)
        cv2.waitKey(1)
        frame += 1
