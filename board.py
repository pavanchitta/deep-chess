import chess


class Board:
    """
    This will wrap a python-chess board to provide a represenation for the board.
    For now, this is very minimal and only functions to contain state, which can
    be fetched and operated on directly.
    """
    def __init__(self, state=None):
        self.state = state if state else chess.Board()

    def white_won(self):
        return self.state.is_game_over() and self.state.result().split('-')[0] == '1'

    def black_won(self):
        return self.state.is_game_over() and self.state.result().split('-')[1] == '1'

    def draw(self):
        return self.state.is_game_over() and self.state.result().split('-')[0] == '1/2'

    def color_won(self, color):
        return self.white_won() if color == chess.WHITE else self.black_won()