#!/usr/bin/python3.5
import argparse
import cv2
from enum import Enum
import sys
import numpy as np

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
    def __init__(self, size):
        self.rows, self.columns = size
        self.squares = []

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

    # draw each sqare
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
                piece_image = piece_atlas.get_piece_image(square.piece)
                piece_image = changeTransparentColorsTo(piece_image, bg_color)

                p_width, p_height = piece_image.shape[1::-1]
                p_p_left, p_p_up = int((square_width - p_width) / 2), int((square_height - p_height) / 2)
                frame[y+p_p_up:y+p_p_up+p_height, x+p_p_left:x+p_p_left+p_width] |= piece_image

    return frame

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

            piece_image = piece_atlas.get_piece_image(Piece(color, type))
            p_width, p_height = piece_image.shape[1::-1]
            p_p_left, p_p_up = int((piece_width - p_width) / 2), int((piece_height - p_height) / 2)

            isSelected = i == selected_piece[0] and j == selected_piece[1]
            bg_color = selected_color if isSelected else background_color

            frame[y:y+piece_height, x:x+piece_width] = bg_color
            piece_image = changeTransparentColorsTo(piece_image, bg_color)
            frame[y+p_p_up:y+piece_height-p_p_up, x+p_p_left:x+piece_width-p_p_left] = piece_image

    return frame


def redrawGUI(
        chessboard,
        piece_atlas,
        selected_piece=(-1, -1), selected_square=(-1, -1),
        board_size = (640, 640), piece_sizes=(55, 55),
        board_right_padding = 50, pieces_padding=(10, 10),
        piece_per_line=3
):
    chessboard_frame = drawChessboard(chessboard, piece_atlas, board_size, selected_square)
    piece_selection_frame = drawPieceSelection(piece_atlas, selected_piece, piece_sizes, pieces_padding, piece_per_line)

    cb_width, cb_height = chessboard_frame.shape[1::-1]
    ps_width, ps_height = piece_selection_frame.shape[1::-1]

    frame = np.zeros((max(cb_height, ps_height), cb_width + board_right_padding + ps_width, 4), dtype=np.uint8)

    frame[0:cb_height, 0:cb_width] = chessboard_frame
    frame[0:ps_height, cb_width + board_right_padding:cb_width+board_right_padding+ps_width] = piece_selection_frame

    return frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Given a video extracts the chessboard'
    )
    parser.add_argument('video_path', help='video file to label the chessboard from')
    parser.add_argument('piece_atlas_path', help='the atlas to read piece images from.')
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
    order = [
        PieceType.King, PieceType.Queen, PieceType.Bishop,
        PieceType.Knight, PieceType.Rook, PieceType.Pawn
    ]

    chessboard = Chessboard((8, 8))

    chessboard.setPiece(0, 1, Piece(PieceColor.White, PieceType.Bishop))

    cv2.imshow('result', redrawGUI(chessboard, piece_atlas, (0, 1)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(-1)
    for i in range(0, 2):
        color = PieceColor.White if i == 0 else PieceColor.Black
        for type in order:
            cv2.imshow('piece', piece_atlas.get_piece_image(Piece(color, type)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

