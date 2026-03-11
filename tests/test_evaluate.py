"""Tests for evaluation harness."""
import pytest

from zsy.agents import RandomAgent, HeuristicAgent
from zsy.evaluate import evaluate_agents, evaluate_vs_baselines


class TestEvaluateAgents:
    def test_random_vs_random(self):
        result = evaluate_agents(
            {"A": RandomAgent, "B": RandomAgent, "C": RandomAgent},
            num_games=100,
            num_players=3,
        )
        assert result.num_games == 100
        assert sum(result.wins.values()) == 100
        # Each agent should have some wins
        assert all(result.wins[name] > 0 for name in ["A", "B", "C"])
        # Win rates should sum to ~1
        total_wr = sum(result.win_rates.values())
        assert abs(total_wr - 1.0) < 0.01

    def test_heuristic_vs_random(self):
        result = evaluate_agents(
            {"Heuristic": HeuristicAgent, "Random1": RandomAgent, "Random2": RandomAgent},
            num_games=200,
            num_players=3,
        )
        # Heuristic should win more than random baseline (33%)
        assert result.win_rates["Heuristic"] > 0.30

    def test_elo_ratings_computed(self):
        result = evaluate_agents(
            {"A": RandomAgent, "B": RandomAgent, "C": RandomAgent},
            num_games=50,
            num_players=3,
        )
        assert len(result.elo_ratings) == 3
        for name, elo in result.elo_ratings.items():
            assert 1000 < elo < 2000  # reasonable range

    def test_summary_output(self):
        result = evaluate_agents(
            {"A": RandomAgent, "B": RandomAgent, "C": RandomAgent},
            num_games=30,
            num_players=3,
        )
        summary = result.summary()
        assert "Tournament" in summary
        assert "30 games" in summary

    def test_seat_rotation(self):
        result = evaluate_agents(
            {"A": RandomAgent, "B": RandomAgent, "C": RandomAgent},
            num_games=60,
            num_players=3,
            seat_rotation=True,
        )
        # All agents should have wins (rotation ensures fairness)
        assert all(result.wins[name] > 0 for name in ["A", "B", "C"])


class TestEvaluateVsBaselines:
    def test_random_agent_vs_baselines(self):
        result = evaluate_vs_baselines(
            RandomAgent,
            agent_name="TestAI",
            num_games=100,
            num_players=3,
        )
        assert result.num_games == 100
        assert "TestAI" in result.win_rates
        assert "Random" in result.win_rates
        assert "Heuristic" in result.win_rates
