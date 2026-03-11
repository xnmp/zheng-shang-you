"""Evaluation harness for comparing AI agents.

Provides round-robin tournament functionality with:
- Win Probability (WP) tracking
- Elo rating computation
- Statistical significance via large game counts
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

from zsy.game import Game, Agent

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a single game."""
    winner: int
    finish_order: list[int]


@dataclass
class TournamentResult:
    """Aggregated results from a tournament."""
    num_games: int
    wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    win_rates: dict[str, float] = field(default_factory=dict)
    elo_ratings: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Tournament: {self.num_games} games"]
        lines.append("-" * 40)
        for name in sorted(self.win_rates, key=lambda n: -self.win_rates[n]):
            wr = self.win_rates[name]
            w = self.wins[name]
            elo = self.elo_ratings.get(name, 1500.0)
            lines.append(f"  {name:20s} WR={wr:.3f} ({w}/{self.num_games}) Elo={elo:.0f}")
        return "\n".join(lines)


def evaluate_agents(
    agent_factories: dict[str, type | callable],
    num_games: int = 1000,
    num_players: int = 3,
    num_decks: int | None = None,
    seat_rotation: bool = True,
) -> TournamentResult:
    """Run a tournament between named agents.

    Args:
        agent_factories: Dict of name -> callable that creates an Agent instance.
            Must have exactly num_players entries.
        num_games: Number of games to play.
        num_players: Number of players per game.
        num_decks: Number of decks (auto if None).
        seat_rotation: If True, rotate seat positions each game.

    Returns:
        TournamentResult with win rates and Elo ratings.
    """
    names = list(agent_factories.keys())
    assert len(names) == num_players, f"Need exactly {num_players} agents, got {len(names)}"

    result = TournamentResult(num_games=num_games)
    elo = {name: 1500.0 for name in names}
    K = 32.0  # Elo K-factor

    for game_idx in range(num_games):
        # Rotate seats if enabled
        if seat_rotation:
            offset = game_idx % num_players
            rotated_names = names[offset:] + names[:offset]
        else:
            rotated_names = names

        agents = [agent_factories[name]() for name in rotated_names]
        game = Game(
            num_players=num_players,
            agents=agents,
            num_decks=num_decks,
        )
        winner_idx = game.run()
        winner_name = rotated_names[winner_idx]
        result.wins[winner_name] += 1

        # Update Elo ratings (multi-player: winner beats all others)
        for i, name in enumerate(rotated_names):
            if name == winner_name:
                continue
            # Expected score for winner vs this loser
            e_winner = _expected_score(elo[winner_name], elo[name])
            e_loser = 1.0 - e_winner
            elo[winner_name] += K * (1.0 - e_winner)
            elo[name] += K * (0.0 - e_loser)

    # Compute final stats
    for name in names:
        result.win_rates[name] = result.wins[name] / num_games
    result.elo_ratings = dict(elo)

    return result


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Elo expected score for player A against player B."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def evaluate_vs_baselines(
    agent_factory: callable,
    agent_name: str = "AI",
    num_games: int = 1000,
    num_players: int = 3,
) -> TournamentResult:
    """Evaluate an agent against random and heuristic baselines.

    Runs a 3-way tournament: agent vs random vs heuristic.
    """
    from zsy.agents import RandomAgent, HeuristicAgent

    factories = {
        agent_name: agent_factory,
        "Random": RandomAgent,
        "Heuristic": HeuristicAgent,
    }
    return evaluate_agents(factories, num_games=num_games, num_players=num_players)
