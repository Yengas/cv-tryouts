#!/usr/bin/python3.5
import sys
import cv2
import numpy as np
import utils

# given the internal corners found, extends it to find all the corners
def calculateAllBoardCorners(corners, patternSize):
    rows, columns = patternSize
    nRows, nColumns = rows + 1, columns + 1
    allCorners = []

    # fill with empty arrays
    for i in range(0, nRows):
        for j in range(0, nColumns):
            allCorners.append({ 'tl': None, 'tr': None, 'bl': None, 'br': None })

    # calculate the starting point
    for i in range(0, rows):
        for j in range(0, columns):
            ni, nc = i + 1, j + 1
            cornerIndex = i * columns + j
            currentIndex = ni * nColumns  + nc
            currentNode = allCorners[currentIndex]

            cornerCreationConfiguration = [
                {
                    'key': 'tr',
                    'check': lambda i, j: j < columns - 1,
                    'target': cornerIndex + 1,
                    'calculate': lambda current, get, _, __: current + (current - get(cornerIndex - 1))
                },
                {
                    'key': 'bl',
                    'check': lambda i, j: i < rows - 1,
                    'target': cornerIndex + columns,
                    'calculate': lambda current, get, _, __: current + (current - get(cornerIndex - columns))
                },
                {
                    'key': 'br',
                    'check': lambda i, j: j < columns - 1 and i < rows - 1,
                    'target': cornerIndex + columns + 1,
                    'calculate': lambda current, get, i, j:
                    current + ((current - get(cornerIndex - columns)) if i > 0 else (get(cornerIndex + columns) - current))
                    + ((get(cornerIndex + 1) - current) if j < columns - 1 else (current - get(cornerIndex - 1)))
                }
            ]

            # current row's top left. don't for get to add for the
            currentNode['tl'] = corners[cornerIndex, 0]
            # foreach definition in the cornerCreationConfiguration, calculate the corners
            for cornerCreation in cornerCreationConfiguration:
                key = cornerCreation['key']
                if cornerCreation['check'](i, j):
                    currentNode[key] = corners[cornerCreation['target'], 0]
                else:
                    def get(index):
                        return corners[index, 0]
                    currentNode[key] = cornerCreation['calculate'](corners[cornerIndex, 0], get, i, j)

    for j in range(0, columns):
        currentIndex = (j + nColumns + 1)
        currentNode = allCorners[currentIndex]
        even = allCorners[j]
        next = allCorners[j + 1]

        even['br'] = next['bl'] = currentNode['tl']
        even['tr'] = next['tl'] = currentNode['tl'] - (allCorners[currentIndex + nColumns]['tl'] - currentNode['tl'])

    for i in range(0, rows):
        currentIndex = (i + 1) * nColumns + 1
        currentNode = allCorners[currentIndex]
        up = allCorners[i * nColumns]
        even = allCorners[(i + 1) * nColumns]
        even['tr'] = up['br'] = currentNode['tl']
        even['tl'] = up['bl'] = currentNode['tl'] - (allCorners[currentIndex + 1]['tl'] - currentNode['tl'])

    allCorners[0]['tl'] = allCorners[0]['tr'] - (allCorners[1]['tr'] - allCorners[1]['tl'])
    allCorners[nColumns - 1]['tr'] = allCorners[nColumns - 1]['tl'] + (allCorners[nColumns - 2]['tr'] - allCorners[nColumns - 2]['tl'])
    allCorners[nColumns - 1]['br'] = allCorners[nColumns - 1]['bl'] + (allCorners[nColumns - 2]['br'] - allCorners[nColumns - 2]['bl'])
    lrfIndex = (nRows - 1) * nColumns
    lrf = allCorners[lrfIndex]
    lrf['bl'] = lrf['tl'] + (allCorners[lrfIndex - nColumns]['bl'] - allCorners[lrfIndex - nColumns]['tl'])
    lrf['br'] = lrf['tr'] + (allCorners[lrfIndex - nColumns]['br'] - allCorners[lrfIndex - nColumns]['tr'])

    return allCorners

# given the all corners as dicts of tl, tr, br, bl corners, finds the homography.
def getBoardHomography(allCorners, patternSize, boardSize):
    row, column = patternSize[0] + 1, patternSize[1] + 1
    width, height = boardSize
    widthStep, heightStep = width / float(column), height / float(row)
    imgPoints = []
    objPoints = []

    for i in range(0, row):
        for j in range(0, column):
            imgPoints.append(allCorners[i * column + j]['tl'])
            objPoints.append([ j * widthStep, i * heightStep ])

    for j in range(0, column):
        imgPoints.append(allCorners[(row - 1) * column + j]['bl'])
        objPoints.append([ j * widthStep, row * heightStep ])

    for i in range(0, row):
        imgPoints.append(allCorners[i * row + (column - 1)]['tr'])
        objPoints.append([ column * widthStep, i * heightStep ])

    imgPoints.append(allCorners[(row - 1) * column + (column - 1)]['br'])
    objPoints.append([ column * widthStep, row * heightStep])

    return cv2.findHomography(np.array(imgPoints, dtype=np.float32), np.array(objPoints, dtype=np.float32))
    #imgPoints = [
    #    allCorners[0]['tl'], allCorners[column - 1]['tr'],
    #    allCorners[(row - 1) * column + column - 1]['br'], allCorners[(row - 1) * column]['bl']
    #]
    #objPoints = [(0, 0), (640, 0), (640, 640), (0, 640)]
    #return cv2.findHomography(np.array(imgPoints, dtype=np.float32), np.array(objPoints, dtype=np.float32))

def findChessboardPlacement(
        image, pattern_size, board_size, gray=None,
        find_chessboard_flags = cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS,
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        search_window = (5, 5), zero_zone = (-1, -1)
):
    # make the image gray
    gray = gray.copy() if gray is not None else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply clahe for contrast limiting, reflection removing
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(11,11))
    cl = clahe.apply(gray)

    # find the chessboard on the clahe applied image
    found, corners = cv2.findChessboardCorners(cl, pattern_size)

    if found is False:
        return False, None, None
    # locate the sub pixel accurate corners on the gray image
    corners = cv2.cornerSubPix(gray, corners, search_window, zero_zone, criteria)
    # derive all the corners of the board, given the inner corners
    corners = calculateAllBoardCorners(corners, pattern_size)
    # get board homography
    homography, _ = getBoardHomography(corners, pattern_size, board_size)

    return True, homography, corners

if __name__ == '__main__':
    PROGRAM_DESC = 'Finds the chessboard in the given image, with opencv findchessboard function'

    parser = utils.createArgumentParserWithImage(PROGRAM_DESC)
    args = parser.parse_args()

    image = cv2.imread(args.image_path)

    if image is None:
        print("Couldn't read image from the given file.")
        sys.exit(-1)

    pattern_size, board_size = (7, 7), (640, 640)
    found, homography, corners = findChessboardPlacement(image, pattern_size, board_size)
    if found is False:
        print("Found no complete corners, sorry!")
        sys.exit(-1)

    out = cv2.warpPerspective(image, homography, board_size)
    cv2.imshow('original', image)
    cv2.imshow('result', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
