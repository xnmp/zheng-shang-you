"""Self-play episode generation for DouZero-style DMC training.

Generates complete game episodes recording (state, action, return) tuples
for every decision point. Returns are computed as the episode reward
(+1 for win, -1 for loss) propagated back to each step.

Supports multi-process generation via generate_episodes().
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from zsy.cards import Card, Deck, Rank
from zsy.combinations import CombinationType
from zsy.encoding import encode_state, encode_action, state_dim, action_dim
from zsy.game import Game, GameState, GamePhase
from zsy.network import QNetwork
from zsy.player import Player
from zsy.wildcard import WildcardAssignment


@dataclass
class Transition:
    """A single (state, action, return) tuple from an episode."""
    player_id: int
    state: np.ndarray
    action: np.ndarray
    reward: float  # filled in after episode ends


@dataclass
class Episode:
    """A complete game episode with per-player transitions."""
    transitions: list[Transition] = field(default_factory=list)
    winner: int = -1
    num_players: int = 3


class RLAgent:
    """Agent that uses a Q-network for move selection during self-play.

    Wraps the QNetwork to implement the game's Agent protocol.
    Uses epsilon-greedy exploration.
    """

    def __init__(
        self,
        network: QNetwork,
        num_players: int = 3,
        epsilon: float = 0.01,
    ) -> None:
        self.network = network
        self.num_players = num_players
        self.epsilon = epsilon
        self._last_state: np.ndarray | None = None
        self._last_action: np.ndarray | None = None

    def choose_move(
        self,
        player: Player,
        game_state: GameState,
        moves: list[WildcardAssignment],
    ) -> WildcardAssignment:
        state_vec = encode_state(player, game_state, self.num_players)
        action_vecs = [encode_action(m) for m in moves]

        idx = self.network.best_action(state_vec, action_vecs, self.epsilon)

        self._last_state = state_vec
        self._last_action = action_vecs[idx]

        return moves[idx]


def generate_episode(
    networks: list[QNetwork] | QNetwork,
    num_players: int = 3,
    epsilon: float = 0.01,
) -> Episode:
    """Generate a single self-play episode.

    Args:
        networks: Either a single QNetwork (shared) or one per player.
        num_players: Number of players.
        epsilon: Exploration rate.

    Returns:
        Episode with transitions and computed returns.
    """
    if isinstance(networks, QNetwork):
        nets = [networks] * num_players
    else:
        nets = networks

    agents = [RLAgent(net, num_players, epsilon) for net in nets]
    game = Game(num_players=num_players, agents=agents)
    game.deal()

    episode = Episode(num_players=num_players)
    transitions_by_player: list[list[Transition]] = [[] for _ in range(num_players)]

    while game.phase == GamePhase.PLAYING:
        current = game.current_player
        player = game.players[current]

        if not player.has_cards:
            game._advance_player()
            continue

        # Play turn (agent records state/action internally)
        record = game.play_turn()
        if record is None:
            break

        agent = agents[current]
        if agent._last_state is not None:
            t = Transition(
                player_id=current,
                state=agent._last_state,
                action=agent._last_action,
                reward=0.0,  # filled in below
            )
            transitions_by_player[current].append(t)
            episode.transitions.append(t)

    # Compute returns: +1 for winner, -1 for losers
    episode.winner = game.winner if game.winner is not None else 0
    for pid in range(num_players):
        reward = 1.0 if pid == episode.winner else -1.0
        for t in transitions_by_player[pid]:
            t.reward = reward

    return episode


def generate_episodes(
    networks: list[QNetwork] | QNetwork,
    num_episodes: int,
    num_players: int = 3,
    epsilon: float = 0.01,
) -> list[Episode]:
    """Generate multiple self-play episodes (single-process).

    For multi-process, call generate_episode() in separate processes.
    """
    return [
        generate_episode(networks, num_players, epsilon)
        for _ in range(num_episodes)
    ]
