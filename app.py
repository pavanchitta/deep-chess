# app.py
import chess
import chess.svg
from run_game import Game
from computer import Computer
from flask import Flask, render_template, request, Response
import time
import keras
from tensorflow.python.keras.backend import set_session
from models.neural_net import NeuralNet

import tensorflow as tf

sess = keras.backend.get_session()
graph = tf.get_default_graph()
set_session(sess)

ml_model_file ='test_10000_bsize64_epochs50'
ml_model = keras.models.load_model(f'models/trained_models/{ml_model_file}')
model = NeuralNet()
model.model = ml_model

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def create_chess():
    print(game.board.state.legal_moves)
    return '''
        <img src="/board">
        <p><form action="/move"><input type=text name=move><input type=submit value="Make move"></form></p>
        <p><form action="/reset"><input type=submit value="Reset"></form></p>
        <p><form action="/undo"><input type=submit value="Undo Move"></form></p>
        <p> Latest computer move: {} </p>
    '''.format(game.board.state.peek().uci() if game.board.state.move_stack else "Null")


@app.route("/reset")
def reset():
    print("Resetting game")
    game.restart_game()
    return create_chess()


@app.route("/undo")
def undo():
    print("Undo latest move")
    assert(len(game.board.state.move_stack) >= 2)
    game.board.state.pop()
    game.board.state.pop()
    return create_chess()


@app.route("/move")
def move():

    global graph
    global sess
    global model
    with graph.as_default():
        set_session(sess)
        move_str = request.args.get('move', default="")
        if (len(move_str) != 4 and len(move_str) != 5):
            return create_chess()
        move = chess.Move.from_uci(move_str)
        if move in game.board.state.legal_moves:
            game.board.state.push(move)
            # Get computer move
            cpu_move = game.computer.generate_move(game.board, model)
            game.board.state.push(cpu_move)
        return create_chess()


@app.route("/board")
def board():
    return Response(chess.svg.board(game.board.state, size=350), mimetype='image/svg+xml')

if __name__ == "__main__":
    # player_is_white = input("Enter: w for White, b for Black: ") == 'w'
    game = Game(player_is_white=True)
    app.run(debug=True)
