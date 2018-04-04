#!/usr/bin/python3.5
import argparse
import sys
import cv2
import numpy as np

# define global variables to be used
POINT_SELECT_WINDOW_NAME = 'Select Points...'
RESULT_WINDOW_NAME = 'Result'
POINTS_TO_SELECT = 4
RESULT_WIDTH, RESULT_HEIGHT = 640, 640

# Argument parsing for image to be selected
parser = argparse.ArgumentParser(
    description="Warps the %d point clicked on the image into a flat rectangle." % POINTS_TO_SELECT
)
parser.add_argument('image_path', help='image file to process')
args = parser.parse_args()

image_path = args.image_path
image = cv2.imread(image_path)

if image is None:
    print("Couldn't read the image from the given path.")
    sys.exit(-1)

points = []
image_to_draw = image.copy()

# draw the given points to the clone of the given image
def draw_points(image, points):
    clone = image.copy()
    for point in points:
        cv2.circle(clone, point, 10, (0, 255, 0))
    return clone

# handle click events to the frames
def on_image_click(event, x, y, flags, param):
    global points, image_to_draw
    # if not double click, or there aren't any more points to select...
    if event != cv2.EVENT_LBUTTONDBLCLK or len(points) >= POINTS_TO_SELECT:
        return False

    # add the clicked position to points
    points.append((x, y))
    # replace the image to be drawn with the changed point array
    image_to_draw = draw_points(image, points)
    return True

print("Please select the corners clock-wise from the top left.")
cv2.namedWindow(POINT_SELECT_WINDOW_NAME)
cv2.setMouseCallback(POINT_SELECT_WINDOW_NAME, on_image_click)
# Point collecting phase
while True:
    cv2.imshow(POINT_SELECT_WINDOW_NAME, image_to_draw)
    key = cv2.waitKey(20) & 0xFF

    if key == 27:
        print("Quiting the app, because escape was pressed.")
        sys.exit(-1)
    # undo if the r button was pressed.
    elif key == ord('r') and len(points) > 0:
        points = points[:-1]
        image_to_draw = draw_points(image, points)

    if len(points) == POINTS_TO_SELECT:
        print("Got enough points!")
        break
cv2.destroyAllWindows()

# where the clicked points should have been, in the ideal world. from top-left to clockwise bottom-left.
dest_points = [(0, 0), (RESULT_WIDTH, 0), (RESULT_WIDTH, RESULT_HEIGHT), (0, RESULT_HEIGHT)]
# create the output image, just trying it out, instead of letting the warp perspective create one.
output_image = np.zeros((RESULT_HEIGHT, RESULT_WIDTH, 3), dtype=np.uint8)
# calculate the homography, given the points clicked on the image, and the destination points.
homography, _ = cv2.findHomography(np.array(points), np.array(dest_points))

# warp the perspective of the original image, so the snippet we want, ends up in the boundry of output_image
cv2.warpPerspective(image, homography, (output_image.shape[1], output_image.shape[0]), output_image)

# show the resulting image, in a new window.
cv2.imshow(RESULT_WINDOW_NAME, output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
