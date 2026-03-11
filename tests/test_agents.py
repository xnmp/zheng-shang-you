"""Tests for built-in agents."""
import random
from zsy.agents import RandomAgent
from zsy.game import Game


class TestRandomAgent:
    def test_runs_full_game(self):
        random.seed(42)
        agents = [RandomAgent() for _ in range(3)]
        game = Game(3, agents)
        winner = game.run()
        assert 0 <= winner < 3

    def test_four_player_game(self):
        random.seed(123)
        agents = [RandomAgent() for _ in range(4)]
        game = Game(4, agents)
        winner = game.run()
        assert 0 <= winner < 4

    def test_multiple_seeds_stable(self):
        for seed in range(10):
            random.seed(seed)
            agents = [RandomAgent() for _ in range(3)]
            game = Game(3, agents)
            winner = game.run()
            assert 0 <= winner < 3
