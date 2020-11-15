####
# Script to process data
####
import os
import numpy as np
import chess.pgn
import time
import sys
from tqdm import tqdm
import random
import re
import git
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.append(f'{homedir}')

PGN_FILEPATH = '/Users/pavanchitta/deep-chess/data/pgn'

CHUNK_SIZE = 100000  # To prevent memory overflow, store files in CHUNK_SIZE


def process_result(res):
    if res.split('-')[0] == '1':
        return 1
    elif res.split('-')[1] == '1':
        return 0
    return 0.5


def preprocess_data(pgn_file_path=PGN_FILEPATH, ngames=-1, use_cache=False, name='', use_chunks=False):
    """
    Load the data and process it to feed to ML models
    Args:
    pgn_file_path: Path to the pgn file containing all the games
    ngames: Limit on number of games to load data for
    """
    if use_cache:
        cached_data_files = list(os.listdir(f'{homedir}/data'))
        if (f'data{name}_{int(ngames)}_games.npz' in cached_data_files):
            print('===== Loading data from cache ======')
            data = np.load(f'{homedir}/data/data{name}_{int(ngames)}_games.npz')
            X = data['X']
            y = data['y']
            return X, y, sample_weights
    X = []
    y = []
    n = 1
    chunk_idx = 0
    print(f"==== Loading {ngames} games worth of Data ======")

    with tqdm(total=100, file=sys.stdout) as pbar:
        for pgn_file in os.listdir(pgn_file_path):
            if not pgn_file.endswith('.pgn'):
                continue
            pgn = open(os.path.join(pgn_file_path, pgn_file))
            try:
                game = chess.pgn.read_game(pgn)
            except Exception as e:
                print(e)
                continue
            board = None
            while game and (ngames == -1 or n <= ngames):
                # if board:
                #     print(board)
                board = game.board()
                
                result = process_result(game.headers['Result'])
                if result == 0.5:
                    game = chess.pgn.read_game(pgn)
                    continue
                num_moves = len(list(game.mainline_moves()))
                # Only sample 10 moves from a single game
                chosen_indices = random.sample(list(range(num_moves))[5:], 10)
                for idx, move in enumerate(game.mainline_moves()):
                    dist = (num_moves - idx) / 2
                    if (board.is_capture(move)):
                        # don't consider states that directly result from a capture
                        # as they can give transient advantage to one side.
                        board.push(move)
                        continue
                    board.push(move)
                    if idx not in chosen_indices:
                        continue
                    X.append(process_board(board))
                    y.append(result)

                game = chess.pgn.read_game(pgn)
                n += 1
                if (n % 10 == 0):
                    pbar.update(10/ngames * 100)

            if (n % int(1e5) == 0) and use_chunks:
                chunk_idx = n // int(1e5)
                np.savez_compressed(open(f"{homedir}/data/data{name}_{int(ngames)}_games_{chunk_idx}.npz", 'wb'),
                         X=X, y=y)
                X = []
                y = []
            if (ngames != -1 and n > ngames):
                break

    print(len(X))
    print(X[-1])

    # Save files for future use
    if len(X) > 0:
        np.savez_compressed(open(f"{homedir}/data/data{name}_{int(ngames)}_games_{chunk_idx+1}.npz", 'wb'),
                 X=X, y=y)

    return np.array(X), np.array(y)


def process_board(board):
    """
    Process a python-chess Board object into a 2D 8x8x1 array representation
    """
    X = np.zeros((773, ))
    piece_dict = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}
    pieces_map = board.piece_map()
    for square, piece in pieces_map.items():
        color = 1 if piece.color == chess.WHITE else 0
        index = piece_dict[piece.symbol().lower()]*128 + 64*color + square
        X[index] = 1

    # Special Indices
    # Move: 768
    SPECIAL_IDX = 768
    X[SPECIAL_IDX] = int(board.turn)
    # Handle Castling rights
    X[SPECIAL_IDX + 1] = int(board.has_kingside_castling_rights(chess.WHITE))
    X[SPECIAL_IDX + 2] = int(board.has_queenside_castling_rights(chess.WHITE))
    X[SPECIAL_IDX + 3] = int(board.has_kingside_castling_rights(chess.BLACK))
    X[SPECIAL_IDX + 4] = int(board.has_queenside_castling_rights(chess.BLACK))
    # Handle Move 
    return X


def load_autoencoder_data(name='', ngames=10000):

    X = []
    pattern = re.compile(f'data{name}_{ngames}_games')
    for file in os.listdir(f'{homedir}/data'):
        match = re.search(pattern, file)
        if match:
            data = np.load(f'{homedir}/data/{file}')
            X.append(data['X'])
    if len(X) == 0:
        raise Exception("No pre-processed data with given parameters")
    return np.concatenate(X, axis=0)


def load_neural_net_data(name='', ngames=100):
    """
    Load data for the Main DeepChess Neural net. Need to generate
    pairs of 
    """
    white_win = []
    white_loss = []
    X = []
    Y = []
    pattern = re.compile(f'data{name}_{ngames}_games')
    for file in os.listdir(f'{homedir}/data'):
        match = re.search(pattern, file)
        if match:
            data = np.load(f'{homedir}/data/{file}')
            tmp_X, tmp_y = data['X'], data['y']

            win_indices = np.argwhere(data['y'] == 1)
            lose_indices = np.argwhere(data['y'] == 0)

            white_win.append(data['X'][win_indices])
            white_loss.append(data['X'][lose_indices])

    # Now generate the 2-tuple train data
    white_win = np.concatenate(white_win)
    white_loss = np.concatenate(white_loss)

    MAX_PAIRS = 5e6
    visited = set()

    while len(X) < MAX_PAIRS:
        i = random.randint(0, white_win.shape[0]-1)
        j = random.randint(0, white_loss.shape[0]-1)
        if (i, j) in visited:
            continue
        else:
            win_game_pos = np.squeeze(white_win[i], axis=0)
            lose_game_pos = np.squeeze(white_loss[j], axis=0)
            order = random.randint(0, 1)
            if order:
                x = np.array([win_game_pos, lose_game_pos])
                y = np.array([1, 0])
            else:
                x = np.array([lose_game_pos, win_game_pos])
                y = np.array([0, 1])
            X.append(x)
            Y.append(y)
            visited.add((i, j))

    return np.array(X), np.array(Y)


if __name__ == "__main__":

    pgn_file = "CCRL-4040.[1190874].pgn"
    preprocess_data(pgn_file_path=PGN_FILEPATH, ngames=100000, use_cache=False, name='', use_chunks=True)

