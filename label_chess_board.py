#!/usr/bin/python3.5
import argparse
import cv2
from enum import Enum
import numpy as np
from os.path import join
import pickle
import sys
import uuid

from find_chess_board import findChessboardPlacement


class PieceType(Enum):
    King = 0
    Queen = 1
    Bishop = 2
    Knight = 3
    Rook = 4
    Pawn = 5

class PieceColor(Enum):
    White = 0
    Black = 1

class SquareColor(Enum):
    White = 0
    Black = 1

class ClickTargetType(Enum):
    Chessboard = 0
    PieceSelection = 1
    Nothing = 2

class BoardStatus(Enum):
    # the board is idle and everything is valid
    Valid = 0
    # the board is not seen, there is another problem etc.
    Invalid = 1
    # wether there is a hand that makes a move on the image etc.
    InProgress = 2

class Piece:
    def __init__(self, color, type):
        self.color = color
        self.type = type

class Square:
    def __init__(self, notation, color, piece = None):
        self.notation = notation
        self.color = color
        self.piece = piece

class Chessboard:
    def __init__(self, size, status = BoardStatus.Valid):
        self.rows, self.columns = size
        self.squares = []
        self.status = status

        for r in range(0, 8):
            square_color = SquareColor.Black if r % 2 == 0 else SquareColor.White
            row = []

            for c in range(0, 8):
                notation = chr(ord('A') + c) + ('%d' % (r + 1))
                row.append(Square(notation, square_color))
                square_color = SquareColor.White if square_color == SquareColor.Black else SquareColor.Black

            self.squares.append(row)

    def setPiece(self, r, c, piece):
        self.squares[r][c].piece = piece

    def setStatus(self, status):
        self.status = status

class PieceAtlas:
    def __init__(self, image, white_first = True, order=None):
        if order == None:
            order = [
                PieceType.King, PieceType.Queen, PieceType.Bishop,
                PieceType.Knight, PieceType.Rook, PieceType.Pawn
            ]
        self.build_piece_map(image, order, white_first)

    def build_piece_map(self, image, order, white_first = True):
        self.pieces = { PieceColor.White: {}, PieceColor.Black: {} }
        width, height = image.shape[1::-1]

        piece_width, piece_height = int(width / len(order)), int(height / 2)
        piece_colors = [PieceColor.White, PieceColor.Black] if white_first else [PieceColor.Black, PieceColor.White]

        for i in range(0, 2):
            for j in range(0, len(order)):
                width_start, height_start = int(piece_width * j), int(piece_height * i)
                piece_image = image[height_start:height_start + piece_height, width_start:width_start + piece_width]
                self.pieces[piece_colors[i]][order[j]] = piece_image

    def get_piece_image(self, piece):
        return self.pieces[piece.color][piece.type]

def changeTransparentColorsTo(frame, color, threshold = 0):
    transparents = frame[:, :, 3] <= threshold
    frame[transparents] = color
    return frame

def drawChessboard(
        chessboard, piece_atlas, board_size = (640, 640), selected_square=(-1, -1),
        white_color = (220, 220, 220, 255), black_color = (47, 79, 79, 255), selected_color=(0, 255, 0, 255)
):
    width, height = board_size
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    # draw checker pattern
    rows, columns = len(chessboard.squares), max(map(len, chessboard.squares))
    square_width, square_height = int(width / columns), int(height / rows)

    # draw each square
    for i in range(0, rows):
        for j in range(0, columns):
            square = chessboard.squares[i][j]
            x, y = int(j * square_width), int((rows - i - 1) * square_height)

            if i == selected_square[0] and j == selected_square[1]:
                bg_color = selected_color
            else:
                bg_color = white_color if square.color == SquareColor.White else black_color

            frame[y:y+square_height, x:x+square_width] = bg_color
            if square.piece is not None:
                piece_image = piece_atlas.get_piece_image(square.piece).copy()
                piece_image = changeTransparentColorsTo(piece_image, bg_color)

                p_width, p_height = piece_image.shape[1::-1]
                p_p_left, p_p_up = int((square_width - p_width) / 2), int((square_height - p_height) / 2)
                frame[y+p_p_up:y+p_p_up+p_height, x+p_p_left:x+p_p_left+p_width] = piece_image

    def cb_resolver(x, y):
        i, j = int((height - y) / square_height), int(x / square_width)
        return (i, j)

    return frame, cb_resolver

def drawPieceSelection(
        piece_atlas, selected_piece = (-1, -1), piece_sizes=(55, 55),
        pieces_padding=(10, 10), piece_per_line=3,
        selected_color=(0, 255, 0, 255), background_color=(220, 220, 220, 255)
):
    total_piece = len(PieceType) * len(PieceColor)
    piece_width, piece_height = piece_sizes
    padding_up, padding_right = pieces_padding
    width = int(((piece_width + padding_right) * (piece_per_line - 1)) + piece_width)
    height = int((((total_piece / piece_per_line) - 1) * (piece_height + padding_up)) + piece_height)
    frame = np.zeros((height, width, 4), dtype=np.uint8)

    for i, color in enumerate(PieceColor):
        for j, type in enumerate(PieceType):
            index = (i * len(PieceType)) + j
            row, column = int(index / piece_per_line), int(index % piece_per_line)
            x, y = column * (piece_width + padding_right), row * (piece_height + padding_up)

            piece_image = piece_atlas.get_piece_image(Piece(color, type)).copy()
            p_width, p_height = piece_image.shape[1::-1]
            p_p_left, p_p_up = int((piece_width - p_width) / 2), int((piece_height - p_height) / 2)

            isSelected = row == selected_piece[0] and column == selected_piece[1]
            bg_color = selected_color if isSelected else background_color

            frame[y:y+piece_height, x:x+piece_width] = bg_color
            piece_image = changeTransparentColorsTo(piece_image, bg_color)
            frame[y+p_p_up:y+piece_height-p_p_up, x+p_p_left:x+piece_width-p_p_left] = piece_image

    def ps_resolver(x, y):
        column = int(x / (piece_width + padding_right))
        row = int(y / (piece_height + padding_up))
        index = (row * piece_per_line) + column
        color = PieceColor(int(index / len(PieceType)))
        type = PieceType(index % len(PieceType))

        return ((row, column), color, type)

    return frame, ps_resolver

def redrawGUI(
        chessboard,
        piece_atlas,
        selected_piece=(-1, -1), selected_square=(-1, -1),
        board_size = (640, 640), piece_sizes=(55, 55),
        board_right_padding = 50, pieces_padding=(10, 10),
        piece_per_line=3
):
    chessboard_frame, cb_resolver = drawChessboard(chessboard, piece_atlas, board_size, selected_square)
    piece_selection_frame, ps_resolver = \
        drawPieceSelection(piece_atlas, selected_piece, piece_sizes, pieces_padding, piece_per_line)

    cb_width, cb_height = chessboard_frame.shape[1::-1]
    ps_width, ps_height = piece_selection_frame.shape[1::-1]
    ps_start_x, ps_start_y = cb_width + board_right_padding, 0

    frame = np.zeros((max(cb_height, ps_height), cb_width + board_right_padding + ps_width, 4), dtype=np.uint8)

    frame[0:cb_height, 0:cb_width] = chessboard_frame
    frame[ps_start_y:ps_start_y+ps_height, ps_start_x:ps_start_x+ps_width] = piece_selection_frame

    # Draw right of the board, below the piece selection
    text_start_x, text_start_y = ps_start_x, ps_start_y + ps_height + 50
    cv2.putText(frame, 'Board Status', (text_start_x, text_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255))
    cv2.putText(frame, str(chessboard.status), (text_start_x, text_start_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 255))


    def mouse_click_resolver(x, y):
        if 0 <= x <= cb_width and 0 <= y <= cb_height:
            return ClickTargetType.Chessboard, cb_resolver(x, y)
        if ps_start_x <= x <= ps_start_x + ps_width and ps_start_y <= y <= ps_start_y + ps_height:
            return ClickTargetType.PieceSelection, ps_resolver(x - ps_start_x, y - ps_start_y)

        return ClickTargetType.Nothing, ()

    return frame, mouse_click_resolver

piece_selected_rc, piece_selected = (-1, -1), None

def create_click_listener(chessboard, resolver):
    def click_listener(event, x, y, flags, param):
        global piece_selected_rc, piece_selected
        if event == cv2.EVENT_LBUTTONDOWN:
            type, data = resolver(x, y)

            if type == ClickTargetType.PieceSelection:
                piece_selected_rc = data[0]
                piece_selected = Piece(data[1], data[2])

            if type == ClickTargetType.Chessboard:
                i, j = data
                square = chessboard.squares[i][j]
                if square.piece == None and piece_selected is not None:
                    chessboard.setPiece(i, j, piece_selected)
                else:
                    chessboard.setPiece(i, j, None)

    return click_listener

def persist_label(frame, chessboard, output_path):
    persist_name = str(uuid.uuid4())
    file_path = join(output_path, persist_name)

    cv2.imwrite(file_path + '.jpg', frame)
    pickle.dump(chessboard, open(file_path + '.obj', 'wb'))
    return persist_name

# Rotates the given homography from the origin point
def getRotatedHomography(homography, size, angle):
    width, height = size
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotationHomography = np.concatenate((rotation, np.array([[0, 0, 1]])))
    return np.dot(rotationHomography, homography)


if __name__ == '__main__':
    BOARD_LABELING_WINDOW_NAME = 'Board Labeling'
    CURRENT_FRAME_WINDOW_NAME = 'Current Frame'
    CURRENT_BOARD_WINDOW_NAME = 'Currernt Frame Board'
    parser = argparse.ArgumentParser(
        description='Given a video extracts the chessboard'
    )
    parser.add_argument('video_path', help='video file to label the chessboard from')
    parser.add_argument('piece_atlas_path', help='the atlas to read piece images from.')
    parser.add_argument('output_path', help='where to save the board and its label.')
    parser.add_argument('--cf', help='whether to show the current frame or not.', default=False, action='store_true')
    args = parser.parse_args()

    video = cv2.VideoCapture(args.video_path)# change to (0) for camera
    width, height, fps = [round(video.get(p)) for p in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS]]
    if video.isOpened() is False:
        print('Could not open the given video file for reading.')
        sys.exit(-1)

    piece_atlas_image = cv2.imread(args.piece_atlas_path, cv2.IMREAD_UNCHANGED)
    if piece_atlas_image is None:
        print('Could not read the piece atlas from the given path.')
        sys.exit(-1)

    piece_atlas = PieceAtlas(piece_atlas_image)
    chessboard = Chessboard((8, 8))
    last_frame, last_board = None, None

    pattern_size, board_size = (7, 7), (640, 640)
    last_homography, last_corners = None, None

    print('Press y when you find a homography that works. To skip, press any other key.')
    # lets first find the homography
    while True:
        ret, frame = video.read()
        if ret is False:
            break
        found, homography, corners = findChessboardPlacement(frame, pattern_size, board_size, None, cv2.CALIB_CB_FAST_CHECK)
        if found:
            board = cv2.warpPerspective(frame, homography, board_size)
            cv2.imshow('homography result', board)
            key = cv2.waitKey(24) & 0xff

            if key == ord('y'):
                last_homography, last_corners = homography, corners
                break

    cv2.destroyAllWindows()
    if last_homography is None:
        print('Could not found the last camera homography.')
        sys.exit(-1)
    print('Found and set homography successfully!')

    # frame skipping keys, increments in the power of tens
    frame_skip_keys = list(map(ord, ['q', 'w', 'e']))

    while True:
        gui, resolver = redrawGUI(chessboard, piece_atlas, piece_selected_rc)
        if last_frame is None:
            ret, last_frame = video.read()
            if ret is False:
                break
            last_board = cv2.warpPerspective(last_frame, last_homography, board_size)
        frame, board = last_frame, last_board

        cv2.namedWindow(BOARD_LABELING_WINDOW_NAME)
        cv2.setMouseCallback(BOARD_LABELING_WINDOW_NAME, create_click_listener(chessboard, resolver))
        cv2.imshow(BOARD_LABELING_WINDOW_NAME, gui)
        cv2.imshow(CURRENT_BOARD_WINDOW_NAME, board)
        if args.cf:
            cv2.imshow(CURRENT_FRAME_WINDOW_NAME, frame)
        key = cv2.waitKey(100) & 0xff

        if key == ord('s'):
            persist_name = persist_label(board, chessboard, args.output_path)
            print('Persisted labeled frame. Name: `%s`' % (persist_name))
        elif key == ord('c'):
            newStatus = BoardStatus((chessboard.status.value + 1) % len(BoardStatus))
            chessboard.setStatus(newStatus)
        elif key == ord('r'):
            last_homography = getRotatedHomography(last_homography, board_size, 90)
            last_board = cv2.warpPerspective(last_frame, last_homography, board_size)
        elif key in frame_skip_keys:
            index = frame_skip_keys.index(key)
            frame_increment_count = 10**index

            for i in range(0, frame_increment_count):
                video.grab()
            last_frame = None
        elif key == 27:
            break

