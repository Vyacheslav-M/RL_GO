import argparse

import h5py

from dlgo import scoring
from dlgo import zero
from dlgo.goboard_fast import GameState, Player, Point

import Net 

def simulate_game(
        board_size,
        black_agent, black_collector,
        white_agent, white_collector):
    print('Starting the game!')
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }

    black_collector.begin_episode()
    white_collector.begin_episode()
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)

    game_result = scoring.compute_game_result(game)
    print(game_result)
    # Give the reward to the right agent.
    if game_result.winner == Player.black:
        black_collector.complete_episode(1)
        white_collector.complete_episode(-1)
    else:
        black_collector.complete_episode(-1)
        white_collector.complete_episode(1)


def main():
    board_size = 9
    encoder = zero.ZeroEncoder(board_size)
    
    model = Net.Model(encoder)   

    black_agent = zero.ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)
    white_agent = zero.ZeroAgent(model, encoder, rounds_per_move=10, c=2.0)

    c1 = zero.ZeroExperienceCollector()
    c2 = zero.ZeroExperienceCollector()
    black_agent.set_collector(c1)
    white_agent.set_collector(c2)

    for i in range(5):
            simulate_game(board_size, black_agent, c1, white_agent, c2)

    exp = zero.combine_experience([c1, c2])
    black_agent.train(exp, 0.01, 2048)


if __name__ == '__main__':
    main()

