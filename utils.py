import json
import numpy as np
import argparse

def readJSONFile(file_path):
    with open(file_path) as json_data:
        return json.load(json_data)

def readCalibration(file_path):
    data = readJSONFile(file_path)
    mtx = np.asarray(data['camera_matrix'], dtype=np.float32)
    dist = np.asarray(data['dist_coefs'], dtype=np.float32)
    return mtx, dist, data['rms']

# Convert polar representation of a line to pair of points
def convertRhoThetaToCoordinates(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    return ((x1, y1), (x2, y2))

def createArgumentParserWithImage(
        program_desc,
        image_parameter_name = 'image_path',
        image_parameter_desc = 'The image file to process'
):
    parser = argparse.ArgumentParser(description=program_desc)
    parser.add_argument(image_parameter_name, help=image_parameter_desc)
    return parser
