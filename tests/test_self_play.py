"""Tests for self-play episode generation."""
import numpy as np
import pytest

from zsy.encoding import state_dim, action_dim
from zsy.network import QNetwork
from zsy.self_play import (
    Episode,
    RLAgent,
    Transition,
    generate_episode,
    generate_episodes,
)


class TestTransition:
    def test_fields(self):
        t = Transition(
            player_id=0,
            state=np.zeros(10),
            action=np.zeros(5),
            reward=1.0,
        )
        assert t.player_id == 0
        assert t.reward == 1.0


class TestGenerateEpisode:
    def test_episode_completes(self):
        net = QNetwork(num_players=3)
        ep = generate_episode(net, num_players=3, epsilon=1.0)
        assert ep.winner >= 0
        assert ep.winner < 3
        assert len(ep.transitions) > 0

    def test_episode_has_transitions(self):
        net = QNetwork(num_players=3)
        ep = generate_episode(net, num_players=3, epsilon=1.0)
        # Each transition should have correct dimensions
        s_dim = state_dim(3)
        a_dim = action_dim()
        for t in ep.transitions:
            assert len(t.state) == s_dim
            assert len(t.action) == a_dim
            assert t.reward in (1.0, -1.0)

    def test_exactly_one_winner(self):
        net = QNetwork(num_players=3)
        ep = generate_episode(net, num_players=3, epsilon=1.0)
        winner_transitions = [t for t in ep.transitions if t.reward == 1.0]
        loser_transitions = [t for t in ep.transitions if t.reward == -1.0]
        # Winner's transitions all have reward 1
        winner_ids = {t.player_id for t in winner_transitions}
        assert len(winner_ids) == 1
        assert ep.winner in winner_ids

    def test_episode_with_separate_networks(self):
        nets = [QNetwork(num_players=3) for _ in range(3)]
        ep = generate_episode(nets, num_players=3, epsilon=1.0)
        assert ep.winner >= 0

    def test_multiple_episodes(self):
        net = QNetwork(num_players=3)
        episodes = generate_episodes(net, num_episodes=3, num_players=3, epsilon=1.0)
        assert len(episodes) == 3
        for ep in episodes:
            assert ep.winner >= 0
            assert len(ep.transitions) > 0


class TestRLAgent:
    def test_choose_move_records_state(self):
        from zsy.game import Game
        net = QNetwork(num_players=3)
        agent = RLAgent(net, num_players=3, epsilon=1.0)
        agents = [agent, RLAgent(net, 3, 1.0), RLAgent(net, 3, 1.0)]
        game = Game(num_players=3, agents=agents)
        game.deal()
        # Play one turn
        game.play_turn()
        # The agent that played should have recorded state
        played_agent = agents[0]  # first player
        assert played_agent._last_state is not None or agents[1]._last_state is not None
