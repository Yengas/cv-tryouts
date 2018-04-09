#!/usr/bin/python3.5
import argparse
import cv2
from enum import Enum
import sys

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
    def __init__(self, notation, square_color, piece = None):
        self.notation = notation
        self.square_color = square_color
        self.piece = piece

class ChessboardLabeled:
    def __init__(self, size):
        self.rows, self.columns = size
        self.squares = []

        for r in range(0, 8):
            square_color = SquareColor.Black if r % 2 == 0 else SquareColor.White
            row = []

            for c in range(0, 8):
                notation = chr(ord('A') + c) + ('%d' % (r + 1))
                row.append(Square(notation, square_color))
                square_color = SquareColor.White if square_color == SquareColor.Black else SquareColor.White

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
                print(height_start, height_start + piece_height, width_start, width_start + piece_width)
                piece_image = image[height_start:height_start + piece_height, width_start:width_start + piece_width]
                self.pieces[piece_colors[i]][order[j]] = piece_image

    def get_piece_image(self, piece):
        return self.pieces[piece.color][piece.type]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Given a video extracts the chessboard'
    )
    parser.add_argument('video_path', help='video file to label the chessboard from')
    parser.add_argument('piece_atlas_path', help='the atlas to read piece images from.', default='./data/Chess_Pieces_Sprite.png')
    args = parser.parse_args()

    video = cv2.VideoCapture(args.video_path)# change to (0) for camera
    width, height, fps = [round(video.get(p)) for p in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS]]
    if video.isOpened() is False:
        print('Could not open the given video file for reading.')
        sys.exit(-1)

    piece_atlas_image = cv2.imread(args.piece_atlas_path)
    if piece_atlas_image is None:
        print('Could not read the piece atlas from the given path.')
        sys.exit(-1)

    piece_atlas = PieceAtlas(piece_atlas_image)
    order = [
        PieceType.King, PieceType.Queen, PieceType.Bishop,
        PieceType.Knight, PieceType.Rook, PieceType.Pawn
    ]
    for i in range(0, 2):
        color = PieceColor.White if i == 0 else PieceColor.Black
        for type in order:
            cv2.imshow('piece', piece_atlas.get_piece_image(Piece(color, type)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

