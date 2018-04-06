#!/usr/bin/python3.5
# implementing the algorithm described in the CVChess by Jay Hack and Prithvi Ramakrishnan
import sys
import cv2
import numpy as np
from sklearn.externals import joblib
import utils

# copied from cvchess
def euclidean_distance (p1, p2):
    """
        Function: euclidean_distance
        ----------------------------
        given two points as 2-tuples, returns euclidean distance
        between them
    """
    assert ((len(p1) == len(p2)) and (len(p1) == 2))
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# copied from cvchess
def get_centroid (points):
    """
        Function: get_centroid
        ----------------------
        given a list of points represented
        as 2-tuples, returns their centroid
    """
    return (np.mean([p[0] for p in points]), np.mean([p[1] for p in points]))

# copied from cvchess
def cluster_points (points, cluster_dist=7):
    """
        Function: cluster_points
        ------------------------
        given a list of points and the distance between them for a cluster,
        this will return a list of points with clusters compressed
        to their centroid
    """
    #=====[ Step 1: init old/new points	]=====
    old_points = np.array (points)
    new_points = []

    #=====[ ITERATE OVER OLD_POINTS	]=====
    while len(old_points) > 1:
        p1 = old_points [0]
        distances = np.array([euclidean_distance (p1, p2) for p2 in old_points])
        idx = (distances < cluster_dist)
        points_cluster = old_points[idx]
        centroid = get_centroid (points_cluster)
        new_points.append (cv2.KeyPoint(centroid[0], centroid[1], 3))
        old_points = old_points[np.invert(idx)]

    return new_points

parser = utils.createArgumentParserWithImage('Finds a chessboard from the given image.')
parser.add_argument(
    '--classifier',
    default='./data/corner-classifier-rf.pkl',
    help='Path to the corner_classifier to be used while eleminating sift descriptors.'
)
args = parser.parse_args()

image_path = args.image_path
corner_classifier = joblib.load(args.classifier)
image = cv2.imread(image_path)

if image is None:
    print("Couldn't read the given image file.")
    sys.exit(-1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hc = cv2.cornerHarris(gray, 2, 3, 0.04)
print(hc)
#hc_kps = hc.detect(image)

#cv2.drawKeypoints(image, hc_kps, out)

#result is dilated for marking the corners, not important
dst = cv2.dilate(hc,None)
dst = dst > (0.02 * dst.max())
width, height = dst.shape[1::-1]
points = []

for i in range(0, height):
    for j in range(0, width):
        if dst[i][j]:
            points.append((j, i))
points = cluster_points(points)
sfd = cv2.xfeatures2d.SIFT_create()
sd = sfd.compute(image, points)[1]

predictions = corner_classifier.predict(sd)
# there are no chessboard corners predicted because the keypoints decided HarrisLaplaceFeatureDetector doesn't land on
# the chessboard.
idx = (predictions == 1)
chessboard_corners = [c for c, i in zip(points, idx) if i == 1]

out = image.copy()
cv2.drawKeypoints(image, chessboard_corners, out)

cv2.imshow('result', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
