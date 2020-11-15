import chess
import models.data_processing as dp
import keras
import numpy as np

class Computer:

    def __init__(self, color, minmax_depth=3):
        self.color = color
        self.depth = minmax_depth

    def value_board(self, board, color):
        """
        Compute the value of a board state for the computer. This will use
        heuristics for piece value and board layout. Ultimately, want to
        incorporate a learned value function as well.
        """
        #print("EVALUATING BOARD")
        if board.state.is_game_over():
            if (board.color_won(color)):
                return float("inf")
            return float("-inf")

        score = 0
        piece_value_dict = {'p': 10, 'n': 30, 'b': 30, 'r': 50, 'q': 90, 'k': 900}
        pieces_map = board.state.piece_map()
        for piece in pieces_map.values():
            piece_value = piece_value_dict[piece.symbol().lower()] * (1 if piece.color == color else -1)
            score += piece_value
        #print("BOARD SCORE: ", score)
        return score

    def value_pawn_advance(self, board, color):

        score = 0
        pieces_map = board.state.piece_map()
        for square, piece in pieces_map.items():
            # First get the penalties on the pieces that are being attacked
            if piece.color == color and piece.symbol().lower() == 'p':
                if piece.color == chess.WHITE:
                    row_adv = square // 8 - 1
                else:
                    row_adv = (63 - square) // 8 - 1
                score += row_adv
        #print(board.state)
        #print("Pawn advance score:", score)
        return score/(6 * 8) # Normalize by the max amount of advancement possible

    def value_attacks(self, board, color):

        # TODO: NEED TO TAKE INTO ACCOUNT WHOSE TURN IT IS

        # TODO: Need to figure out a good way to normalize this score to the
        # same scale as the other scores so the weights are interpretable.

        attacked_score = []
        attacking_score = []
        max_attacked = 0
        max_attacking = 0
        piece_value_dict = {'p': 10, 'n': 30, 'b': 30, 'r': 50, 'q': 90, 'k': 0}

        future_attack_discount = 0.2

        # First get the penalties on the pieces that are being attacked
        pieces_map = board.state.piece_map()
        for square, piece in pieces_map.items():
            # First get the penalties on the pieces that are being attacked
            if piece.color == color:
                attackers = board.state.attackers(not color, square)
                protectors = board.state.attackers(color, square)
                piece_value = piece_value_dict[piece.symbol().lower()]
                if len(list(attackers)) > len(list(protectors)):
                    attacked_score.append(piece_value)
                    max_attacked = max(piece_value, max_attacked)
                elif len(attackers) > 0:
                    min_attacker = min([piece_value_dict[board.state.piece_at(square).symbol().lower()] for square in attackers])
                    if min_attacker < piece_value:
                        attacked_score.append(piece_value - min_attacker)
                        max_attacked = max(piece_value - min_attacker, max_attacked)

            # Now consider the reward for own attacking pieces
            if piece.color != color:
                attackers = board.state.attackers(color, square)
                protectors = board.state.attackers(not color, square)
                piece_value = piece_value_dict[piece.symbol().lower()]
                if len(list(attackers)) > len(list(protectors)):
                    attacking_score.append(piece_value)
                    max_attacking = max(piece_value, max_attacking)
                elif len(attackers) > 0:
                    min_attacker = min([piece_value_dict[board.state.piece_at(square).symbol().lower()] for square in attackers])
                    if min_attacker < piece_value:
                        attacking_score.append(piece_value - min_attacker)
                        max_attacking = max(piece_value - min_attacker, max_attacking)

        attacking_score = (1-future_attack_discount)*max_attacking + future_attack_discount * sum(attacking_score)
        attacked_score = (1-future_attack_discount)*max_attacked + future_attack_discount * sum(attacked_score)

        # print("Attacking score:", attacking_score)
        # print("Attacked score:", attacked_score)
        # print(board.state)

        # if board.state.turn == self.color:
        #     return attacking_score - attacked_score
        # else:
        #     return 2*attacking_score - attacked_score
        return -attacked_score

    def value_board_ML(self, board, color, model, beta=0.1):
        """
        Compute the value of a board state for the computer. This will use a linear
        interpolation between score given by heuristics and the score given by
        the ML model.

        beta is the relative weights given to the ML model vs the heuritics
        """

        #print("EVALUATING BOARD WITH ML AND HEURISTICS WITH BETA = ", beta)
        if board.state.is_game_over():
            if (board.color_won(color)):
                return float("inf")
            return float("-inf")

        max_heuristics_score = 8*10 + 4*30 + 2*50 + 90
        overall_attack_score = self.value_attacks(board, color)
        heuristics_score = (self.value_board(board, color) + max_heuristics_score + overall_attack_score) / (2*max_heuristics_score)
        X = dp.process_board_channels_w_turn(board.state)
        #print(X.shape)
        ml_score = model.predict(np.expand_dims(X, axis=0)).flatten()[0]
        if color == chess.BLACK:
            ml_score = 1 - ml_score

        if board.state.fullmove_number < 5:
            beta = 0.1

        # Pawn advancement score factor
        gamma = 0.05

        score = beta * ml_score + (1-beta)*heuristics_score + \
                gamma * self.value_pawn_advance(board, color)

        score /= (1 + gamma)

        #print("BOARD SCORE: ", score)
        return score

    def generate_move(self, board, model=None):
        """
        Use a minimax algorithm in combination with the value function to
        generate the next move for the computer. Return this move
        """
        def minimax(board, depth, alpha_pos, beta_pos, cpu_turn, moves):

            if depth == 0 or board.state.is_game_over():
                # when depth is odd, cpu_turn will be false in the last stage
                # even though we want to optimize the board for cpu at the leaf
                return np.expand_dims(dp.process_board(board.state), axis=0), moves
            if cpu_turn:
                # Want to maximize value
                max_score_pos = None
                best_move = list(board.state.legal_moves)[0]
                best_moves = []
                for move in board.state.legal_moves:
                    board.state.push(move)
                    pos, new_moves = minimax(board, depth-1, alpha_pos, beta_pos, not cpu_turn, moves + [move.uci()])
                    if max_score_pos is None:
                        max_score_pos = pos
                    else:
                        print(pos.shape)
                        print(max_score_pos.shape)
                        res = model.predict([pos, max_score_pos])
                        print("======")
                        print("Res: ", res)
                        print("boards for above res:\n")
                        print(board.state)
                        # Assume CPU is always black
                        if res < 0.5:
                            best_move = move
                            max_score_pos = pos
                            best_moves = new_moves
                    if alpha_pos is None or model.predict([max_score_pos, alpha_pos]) < 0.5:
                        alpha_pos = max_score_pos
                    board.state.pop()
                    if beta_pos is not None:
                        res = model.predict([alpha_pos, beta_pos])
                        if res < 0.5:
                            break
                # Need to return the best move as well if in the outer most depth
                return (max_score_pos, best_moves) if depth != self.depth else (max_score_pos, best_moves, best_move)

            else:
                # Assume optimal play by opponent to minimize value
                min_score_pos = None
                best_moves = []
                for move in board.state.legal_moves:
                    board.state.push(move)
                    pos, new_moves = minimax(board, depth-1, alpha_pos, beta_pos, not cpu_turn, moves + [move.uci()])
                    if min_score_pos is None:
                        min_score_pos = pos
                    else:
                        res = model.predict([pos, min_score_pos])
                        if res > 0.5:
                            min_score_pos = pos
                            best_moves = new_moves
                    board.state.pop()
                    if beta_pos is None or model.predict([min_score_pos, beta_pos]) > 0.5:
                        beta_pos = min_score_pos
                    if alpha_pos is not None and beta_pos is not None:
                        res = model.predict([alpha_pos, beta_pos])
                        if res > 0.5:
                            break
                return min_score_pos, best_moves

        if board.state.is_game_over():
            return chess.Move.null()
        max_score, best_moves, best_move = minimax(board, self.depth, None, None, True, [])
        print("Best Computer Move score: ", max_score)
        print("Anticipated Best Move sequence: ", best_moves)
        assert(best_move != None)
        return best_move
