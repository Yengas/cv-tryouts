#!/usr/bin/python3.5
import sys
import cv2

# if no output path is given, give an error
if len(sys.argv) < 2:
    print("To capture a video, run `./save_video.py <output_path>`")
    sys.exit(-1)

output_path = sys.argv[1]
# get the camera stream
cap = cv2.VideoCapture(0)

# if couldn't find the camera, exit
if not cap.isOpened():
    print("Couldn't open the video capture.")
    sys.exit(-1)

# get fps, width and height from the opened camera stream
fps, width, height = [int(cap.get(p)) for p in [cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]]

# create a writer with the camera properties and xvid codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# while the camera is open
while cap.isOpened():
    ret, frame = cap.read()
    # if couldn't read, exit
    if ret is False:
        break
    # if the esc key is pressed, exit
    if cv2.waitKey(1) == 27:
        break

    # write the read stream
    writer.write(frame)
    # show the frame
    cv2.imshow('out', frame)

cap.release()
writer.release()
cv2.destroyAllWindows()
