from board import Board
import chess
from computer import Computer
import keras


class Game:

    def __init__(self, player_is_white=True):
        self.board = Board()
        self.player_color = chess.WHITE if player_is_white else chess.BLACK
        # TODO: Implement functionality for player being black
        # if not player_is_white:
        #     self.board.state = self.board.state.mirror()
        self.computer = Computer(not self.player_color)

    def is_over(self):
        return self.board.state.is_game_over()

    def restart_game(self):
        self.board.state.reset()

    def player_to_move(self):
        return self.board.state.turn == self.player_color

ml_model_file ='test_10000_bsize64_epochs50'
ml_model = keras.models.load_model(f'models/trained_models/{ml_model_file}')

def run_game():

    player_is_white = input("Enter: w for White, b for Black: ") == 'w'
    game = Game(player_is_white)
    board = game.board
    while (not game.is_over()):
        if game.player_to_move():
            print(board.state)
            print("legal moves are: ", list(board.state.legal_moves))
            while True:
                move_str = input("Enter move as uci string: ")
                move = chess.Move.from_uci(move_str)
                if move in board.state.legal_moves:
                    break
        else:
            move = game.computer.generate_move(board, ml_model)
        board.state.push(move)

    print("GAME OVER")
    print(f"Result: {game.result()}")


if __name__ == "__main__":
    run_game()
