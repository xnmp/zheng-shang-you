"""Tests for team mode support."""
import random
from zsy.agents import RandomAgent
from zsy.game import Game, GamePhase


class TestTeamMode:
    def test_team_assignment(self):
        agents = [RandomAgent() for _ in range(4)]
        game = Game(4, agents, teams=[0, 1, 0, 1])
        assert game.players[0].team == 0
        assert game.players[1].team == 1
        assert game.players[2].team == 0
        assert game.players[3].team == 1

    def test_winning_team_set(self):
        random.seed(42)
        agents = [RandomAgent() for _ in range(4)]
        game = Game(4, agents, teams=[0, 1, 0, 1])
        winner = game.run()
        assert game.winning_team is not None
        assert game.winning_team == game.teams[winner]

    def test_team_game_completes(self):
        for seed in range(5):
            random.seed(seed)
            agents = [RandomAgent() for _ in range(4)]
            game = Game(4, agents, teams=[0, 1, 0, 1])
            winner = game.run()
            assert game.phase == GamePhase.FINISHED
            assert game.winning_team in (0, 1)

    def test_no_team_mode(self):
        random.seed(42)
        agents = [RandomAgent() for _ in range(3)]
        game = Game(3, agents)
        winner = game.run()
        assert game.teams is None
        assert game.winning_team is None

    def test_teams_in_game_state(self):
        agents = [RandomAgent() for _ in range(4)]
        game = Game(4, agents, teams=[0, 1, 0, 1])
        game.deal()
        state = game.get_state()
        assert state.teams == [0, 1, 0, 1]

    def test_three_team_game(self):
        random.seed(77)
        agents = [RandomAgent() for _ in range(6)]
        game = Game(6, agents, teams=[0, 1, 2, 0, 1, 2])
        winner = game.run()
        assert game.winning_team in (0, 1, 2)
