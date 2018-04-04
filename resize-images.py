#!/usr/bin/python3.5
import sys
import cv2
import os

if len(sys.argv) < 2:
    print("Please run the program as: `./resize-images.py <folder>`")
    sys.exit(-1)

def resize_image(image_path, dest_path):
    # read the image from the given path
    frame = cv2.imread(image_path)
    if frame is None:
        return False
    # generate a temporary file path for the image to be written
    temp_file_path = image_path + '.tmp.jpg'

    # remove the temp file if exists
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    # get original width and height
    width, height = list(map(float, frame.shape[1::-1]))
    # calculate new sizes for either landscape or portrait mode
    newSize = (1280, int(1280 * (height / width))) if width > height else (int(1280 * (width / height)), 1280)

    # resize the image to its new size
    resized_image = cv2.resize(frame, newSize)
    # write out to temp file
    cv2.imwrite(temp_file_path, resized_image)
    # replace the original file with the newly written temp file
    os.replace(temp_file_path, dest_path)
    return True


target_folder = sys.argv[1]

for image_name in os.listdir(target_folder):
    image_path = os.path.join(target_folder, image_name)
    print("Resizinig and overwriting '%s'..." % (image_path))
    # resize and overwrite the image
    done = resize_image(image_path, image_path)
    print("Opeartion: %s." % ("successful" if done else "failed"))
