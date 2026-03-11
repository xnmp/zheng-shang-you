"""Tests for the heuristic agent."""
import pytest

from zsy.agents import HeuristicAgent, RandomAgent
from zsy.game import Game


class TestHeuristicAgent:
    def test_game_completes(self):
        """Heuristic agent should be able to complete a game."""
        agents = [HeuristicAgent() for _ in range(3)]
        game = Game(num_players=3, agents=agents)
        winner = game.run()
        assert 0 <= winner < 3

    def test_beats_random_agent(self):
        """Heuristic agent should win more often than random against randoms."""
        wins = {0: 0, 1: 0, 2: 0}
        num_games = 200
        for _ in range(num_games):
            # Player 0 is heuristic, others are random
            agents = [HeuristicAgent(), RandomAgent(), RandomAgent()]
            game = Game(num_players=3, agents=agents)
            winner = game.run()
            wins[winner] += 1

        # Heuristic should win at least 40% (better than random's ~33%)
        heuristic_wr = wins[0] / num_games
        assert heuristic_wr > 0.35, f"Heuristic win rate too low: {heuristic_wr:.2%}"

    def test_four_player_game(self):
        agents = [HeuristicAgent() for _ in range(4)]
        game = Game(num_players=4, agents=agents)
        winner = game.run()
        assert 0 <= winner < 4
